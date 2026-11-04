"""统一实验执行器：UTF-8 日志、开始/结束时间、真实退出码与旧混码归档。"""
import argparse
import datetime as dt
import os
import subprocess
import sys


def timestamp():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def archive_if_mixed(path):
    """历史日志含 NUL/UTF-16 或不是严格 UTF-8 时，先归档再新建。"""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        raw = f.read()
    mixed = (b'\x00' in raw or raw.startswith((b'\xff\xfe', b'\xfe\xff')))
    if not mixed:
        try:
            raw.decode('utf-8', errors='strict')
        except UnicodeDecodeError:
            mixed = True
    if not mixed:
        return None
    suffix = dt.datetime.now().strftime('%H%M%S%f')[:-3]
    dst = path + '.mixed-' + suffix + '.bak'
    k = 1
    while os.path.exists(dst):
        dst = path + '.mixed-' + suffix + f'-{k}.bak'
        k += 1
    os.replace(path, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--program', required=True)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--log_dir', default='Test-log')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = list(args.command)
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        ap.error('必须在 -- 后提供待执行脚本及参数')

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    os.makedirs(args.log_dir, exist_ok=True)
    day = dt.datetime.now().strftime('%Y%m%d')
    log_path = os.path.join(args.log_dir,
                            f'{args.program}-log-{day}.txt')
    archived = archive_if_mixed(log_path)
    if archived:
        print(f'[日志] 已归档旧混码文件: {archived}', flush=True)

    script = command[0]
    if not os.path.isabs(script):
        script = os.path.join(root, script)
    cmd = [args.python, script] + command[1:]
    shown = subprocess.list2cmdline(cmd)
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    status = 127
    with open(log_path, 'a', encoding='utf-8', newline='') as log:
        def emit(line=''):
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()

        emit('=' * 60)
        if archived:
            emit(f'[日志] 已归档旧混码文件: {archived}')
        emit(f'[开始] {shown}')
        emit(f'[开始时间] {timestamp()}')
        try:
            proc = subprocess.Popen(
                cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                errors='replace', bufsize=1)
            try:
                for line in proc.stdout:
                    emit(line.rstrip('\r\n'))
                status = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                status = 130
                emit('[中断] 收到键盘中断，子进程已终止')
        except OSError as exc:
            emit(f'[启动失败] {exc}')
            status = 127
        emit(f'[结束时间] {timestamp()} (退出码={status})')
        if status != 0:
            emit(f'[错误] {args.program} 失败，详见 {log_path}')
    return int(status)


if __name__ == '__main__':
    sys.exit(main())
