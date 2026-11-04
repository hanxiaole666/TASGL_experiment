"""download_data.py — 全部实验数据集免账号自动下载 + 断点续传

支持数据集 (对应设计方案第三节):
  elliptic     Elliptic (PyG 官方公开镜像, 与 Kaggle 完全一致)
  ethereum     Ethereum Fraud (GitHub 公开镜像, 9841 地址 / 2179 欺诈)
  dgraphfin    DGraph-Fin (信也科技官方 OSS 直链, 370 万节点)
  dexposure    DeXposure (官方 GitHub, 75MB 快速快照; --full 为 1GB 全量)
  bitheteronet BitHeteroNet (作者官方固定文件 ID, 1 主 + 3 备用通道)

特性:
  - HTTP Range 断点续传: 中断后重跑自动从 .part 文件续传
  - 多通道回退 (GitHub raw/代理/media/API; BitHeteroNet 三直链 + gdown)
  - 已下载文件自动跳过; zip 完整性校验; 单个数据集失败不阻塞其他数据集

用法:
  python download_data.py --dataset all            # 全部 (默认)
  python download_data.py --dataset elliptic
  python download_data.py --dataset dexposure --full
"""
import argparse
import base64
import json
import os
import time
import urllib.request
import urllib.error
import zipfile

MIRROR = 'https://data.pyg.org/datasets/elliptic'
ELLIPTIC_FILES = [
    'elliptic_txs_features.csv',
    'elliptic_txs_classes.csv',
    'elliptic_txs_edgelist.csv',
]
CHUNK = 1 << 20          # 1 MB
MAX_RETRY = 8
UA = {'User-Agent': 'TASGL-experiment/1.0'}


# ------------------------------------------------------------------
# 基础下载器 (Range 断点续传)
# ------------------------------------------------------------------
def _human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.1f}{unit}'
        n /= 1024


def _open(url, headers=None, timeout=60):
    h = dict(UA)
    h.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                  timeout=timeout)


def download_with_resume(url, dest, max_retry=MAX_RETRY, timeout=60):
    """带 Range 断点续传的下载器。临时文件 dest+'.part', 完成后原子重命名。
    timeout 同时限制连接与单次读取; BitHeteroNet 用较短值快速切换通道。"""
    part = dest + '.part'
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    for attempt in range(1, max_retry + 1):
        offset = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {}
        if offset > 0:
            headers['Range'] = f'bytes={offset}-'
        try:
            with _open(url, headers, timeout=timeout) as r:
                if offset > 0 and r.status != 206:
                    print('  服务器未接受 Range, 重新下载')
                    offset = 0
                total = r.headers.get('Content-Length')
                total = int(total) + offset if total else None
                mode = 'ab' if offset > 0 else 'wb'
                done, t0 = offset, time.time()
                with open(part, mode) as f:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        print(f'\r  {os.path.basename(dest)}: {_human(done)}'
                              + (f' / {_human(total)}' if total else '')
                              + f'  ({done / max(time.time() - t0, 1e-6) / 1e6:.2f} MB/s)',
                              end='', flush=True)
            print()
            os.replace(part, dest)
            return dest
        except (urllib.error.URLError, ConnectionError, TimeoutError,
                OSError) as e:
            wait = min(2 ** attempt, 30)
            print(f'\n  下载中断 ({str(e)[:100]}); {wait}s 后重试 '
                  f'[{attempt}/{max_retry}], 已保留 {_human(offset)} 断点')
            time.sleep(wait)
    raise RuntimeError(f'下载失败(重试{max_retry}次): {url}')


def download_github_file(repo, path, dest, ref='main'):
    """GitHub 文件多通道下载:
    raw -> ghfast 镜像 -> gh-proxy 镜像 -> media(LFS) -> API base64(重试3次)。
    前四个通道均支持 Range 断点续传; API 通道为一次性下载。"""
    raw_url = f'https://raw.githubusercontent.com/{repo}/{ref}/{path}'
    channels = [
        raw_url,
        f'https://ghfast.top/{raw_url}',
        f'https://gh-proxy.com/{raw_url}',
        f'https://media.githubusercontent.com/media/{repo}/{ref}/{path}',
    ]
    errors = []
    for url in channels:
        try:
            return download_with_resume(url, dest, max_retry=2)
        except RuntimeError as e:
            errors.append(str(e))
            print(f'  通道失败, 切换下一通道: {url}')
    # API 回退 (未认证有速率限制; 网络抖动时重试)
    last_err = None
    for attempt in range(3):
        try:
            print(f'  尝试 GitHub API 通道 (base64, 第{attempt + 1}次)...')
            with _open(f'https://api.github.com/repos/{repo}/contents/{path}?ref={ref}') as r:
                meta = json.load(r)
            with _open(meta['git_url']) as r:
                blob = json.load(r)
            raw = base64.b64decode(blob['content'])
            if len(raw) != meta.get('size', len(raw)):
                raise IOError(f'大小不符 {len(raw)} != {meta["size"]}')
            os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
            with open(dest, 'wb') as f:
                f.write(raw)
            print(f'  API 通道完成: {_human(len(raw))}')
            return dest
        except Exception as e:
            last_err = e
            print(f'  API 通道失败: {str(e)[:120]}')
            time.sleep(3 * (attempt + 1))
    errors.append(str(last_err))
    raise RuntimeError('GitHub 全部通道失败:\n' + '\n'.join(errors))


def fetch_github_text(repo, path, ref='main'):
    """获取 GitHub 小文本文件 (README 等): raw -> jsDelivr -> ghfast -> API。"""
    raw_url = f'https://raw.githubusercontent.com/{repo}/{ref}/{path}'
    for url in (raw_url,
                f'https://cdn.jsdelivr.net/gh/{repo}@{ref}/{path}',
                f'https://ghfast.top/{raw_url}'):
        try:
            with _open(url, timeout=30) as r:
                return r.read().decode('utf-8', errors='replace')
        except Exception:
            continue
    with _open(f'https://api.github.com/repos/{repo}/contents/{path}?ref={ref}') as r:
        meta = json.load(r)
    with _open(meta['git_url']) as r:
        blob = json.load(r)
    return base64.b64decode(blob['content']).decode('utf-8', errors='replace')


def _extract_zip(zip_path, out_dir):
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            os.remove(zip_path)
            raise RuntimeError(f'压缩包损坏已删除, 请重跑: {bad}')
        zf.extractall(out_dir)


# ------------------------------------------------------------------
# 数据集 1: Elliptic
# ------------------------------------------------------------------
def ensure_elliptic(data_dir='./data/elliptic', verbose=True):
    os.makedirs(data_dir, exist_ok=True)
    for name in ELLIPTIC_FILES:
        csv_path = os.path.join(data_dir, name)
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            if verbose:
                print(f'[elliptic] 已存在, 跳过: {csv_path}')
            continue
        zip_path = os.path.join(data_dir, name + '.zip')
        if verbose:
            print(f'[elliptic] 下载 {name} <- {MIRROR}/{name}.zip')
        download_with_resume(f'{MIRROR}/{name}.zip', zip_path)
        _extract_zip(zip_path, data_dir)
        os.remove(zip_path)
        if not os.path.exists(csv_path):   # 兼容 zip 内含子目录
            for root, _, files in os.walk(data_dir):
                if name in files and root != data_dir:
                    os.replace(os.path.join(root, name), csv_path)
        if not os.path.exists(csv_path):
            raise RuntimeError(f'解压后未找到 {csv_path}')
        if verbose:
            print(f'[elliptic] 完成: {csv_path}')
    return data_dir


# ------------------------------------------------------------------
# 数据集 2: Ethereum Fraud (Kaggle 的 GitHub 公开镜像)
# ------------------------------------------------------------------
ETH_REPO = 'JuliaAI/Imbalance.jl'
ETH_PATH = 'docs/src/examples/fraud_detection/transactions.csv'


def ensure_ethereum(data_dir='./data/ethereum', verbose=True):
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, 'transaction_dataset.csv')
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        if verbose:
            print(f'[ethereum] 已存在, 跳过: {dest}')
        return data_dir
    if verbose:
        print('[ethereum] 下载 transaction_dataset.csv '
              '(Kaggle ethereum-frauddetection-dataset 的公开镜像)')
    download_github_file(ETH_REPO, ETH_PATH, dest, ref='dev')
    # 校验: 必须含 FLAG 列
    with open(dest, encoding='utf-8', errors='ignore') as f:
        header = f.readline()
    if 'FLAG' not in header:
        os.remove(dest)
        raise RuntimeError('Ethereum CSV 校验失败(无 FLAG 列), 已删除, 请重跑')
    if verbose:
        print(f'[ethereum] 完成: {dest}')
    return data_dir


# ------------------------------------------------------------------
# 数据集 3: DGraph-Fin (官方 OSS 直链)
# ------------------------------------------------------------------
DGRAPH_URL = 'https://dgraph.oss-cn-shanghai.aliyuncs.com/DGraphFin.zip'


def ensure_dgraphfin(data_dir='./data/dgraphfin', verbose=True):
    os.makedirs(data_dir, exist_ok=True)
    npz = os.path.join(data_dir, 'dgraphfin.npz')
    if os.path.exists(npz):
        if verbose:
            print(f'[dgraphfin] 已存在, 跳过: {npz}')
        return data_dir
    zip_path = os.path.join(data_dir, 'DGraphFin.zip')
    if verbose:
        print(f'[dgraphfin] 下载 DGraphFin.zip <- {DGRAPH_URL} (约 144MB)')
    download_with_resume(DGRAPH_URL, zip_path)
    _extract_zip(zip_path, data_dir)
    if not os.path.exists(npz):            # 兼容 zip 内含子目录
        for root, _, files in os.walk(data_dir):
            if 'dgraphfin.npz' in files:
                os.replace(os.path.join(root, 'dgraphfin.npz'), npz)
    if not os.path.exists(npz):
        raise RuntimeError('解压后未找到 dgraphfin.npz')
    os.remove(zip_path)
    if verbose:
        print(f'[dgraphfin] 完成: {npz}')
    return data_dir


# ------------------------------------------------------------------
# 数据集 4: DeXposure (官方 GitHub 快照)
# ------------------------------------------------------------------
DEX_REPO = 'dthinkr/DeXposure'


def ensure_dexposure(data_dir='./data/dexposure', full=False, verbose=True):
    os.makedirs(data_dir, exist_ok=True)
    name = ('historical-network_week_2020-03-30.json' if full
            else 'historical-network_week_2025-07-01.json')
    dest = os.path.join(data_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        if verbose:
            print(f'[dexposure] 已存在, 跳过: {dest}')
        return data_dir
    if verbose:
        print(f'[dexposure] 下载 {name} '
              f'({"全量 283 周快照, 约 1GB" if full else "快速测试快照, 约 75MB"})')
    download_github_file(DEX_REPO, f'data/{name}', dest, ref='main')
    # 校验: 是 JSON 而非 LFS 指针
    with open(dest, 'rb') as f:
        head = f.read(200)
    if head.startswith(b'version https://git-lfs'):
        os.remove(dest)
        raise RuntimeError('下载到的是 LFS 指针而非数据, 请重跑或改用 --full 之外的模式')
    if verbose:
        print(f'[dexposure] 完成: {dest}')
    return data_dir


# ------------------------------------------------------------------
# 数据集 5: BitHeteroNet (作者官方文件 ID, 1 主 + 3 备用下载通道)
# ------------------------------------------------------------------
BHN_REPO = 'KellyGong/BitHeteroNet'
BHN_DRIVE_URL = ('https://drive.google.com/drive/folders/'
                 '1ZGx0hQYyK7B5rU0G_gGqIu_wbqYbP4wY?usp=sharing')

# 2026-10-26 从作者官方公开文件夹逐页核验的固定文件 ID。
# 文件名与作者 dataloader.py 一致; README 中 addr_classes/tx_features
# 是旧称。使用固定 ID 可跳过最容易超时的文件夹列表解析。
#
# shape 是发布数组布局，不再混同 README 的特征计数。上游 Elliptic++
# 官方论文/教程说明：交易 183 列含 Time step；地址 56 列由 class、
# 28 个交易相关列和 27 个时间相关列（含 Time step）组成。当前发布数组
# 将交易时间步写入 tx_time_step.npy、将地址类别写入 address_classes.npy，
# 因而分别保留 182 和 55 列。BitHeteroNet README 的 183/54 保留为
# documented_shape，用于持续暴露文档与发布布局的不一致。
# 证据边界：上游字段组成支持该布局，但作者未提供转换脚本或逐列映射；
# valid 仅表示物理完整性和此版本布局通过，不代表列语义已独立核验。
BHN_LAYOUT_PROFILE = 'ellipticpp_separated_time_and_label_v1'
BHN_SCHEMA_REFERENCES = {
    'bitheteronet_repository_tree':
        'https://github.com/KellyGong/BitHeteroNet/tree/'
        'cda502c8139209184d78b934bbe98aabdd4949c6',
    'ellipticpp_paper': 'https://doi.org/10.1145/3580305.3599803',
    'ellipticpp_repository':
        'https://github.com/git-disl/EllipticPlusPlus',
}
BHN_FILES = {
    'address_attrs.npy': {
        'id': '1ihbv_1AxhSN8g0uLOL-KY6AW0VO3U5NP',
        'shape': (822942, 55),
        'documented_shape': (822942, 54),
        'layout_note': '55=28 交易相关+27 时间相关；class 单独存放',
        'semantic_status': 'upstream_schema_consistent_column_order_unverified',
    },
    'address_classes.npy': {
        'id': '1_XYSfKf44S9l6oIoGxUS4Z0GY1UJk3cM',
        'shape': (822942,),
    },
    'data.pkl': {
        'id': '1bodrsOJ4fZr0f-JTk6prfYmcr67riqNN',
        'shape': None,
    },
    'tx_attrs.npy': {
        'id': '1dWoCPpV63HP5Vu7pjRIeg7Zd2ZeadhHL',
        'shape': (202804, 182),
        'documented_shape': (202804, 183),
        'layout_note': '182=183 交易列-Time step；时间步单独存放',
        'semantic_status': 'upstream_schema_consistent_column_order_unverified',
    },
    'tx_classes.npy': {
        'id': '1WgIefwMajBEsiTx7zSptco7ARu_fVobF',
        'shape': (202804,),
    },
    'tx_time_step.npy': {
        'id': '13lfbZiL5N0nLHQQ4085koC5ZTcKCrKBF',
        'shape': (202804,),
    },
}
BHN_ALIASES = {
    'addr_classes.npy': 'address_classes.npy',
    'tx_features.npy': 'tx_attrs.npy',
}

# 注意: 这些是同一作者官方数据的不同下载通道, 不是独立第三方镜像。
# 前三个 HTTP 通道已用 tx_classes.npy 实测, SHA-256 完全一致;
# gdown 通道负责大文件确认页及额外重试。任一通道成功后提升为后续文件首选。
BHN_CHANNELS = [
    ('usercontent', 'Google UserContent 直链',
     'https://drive.usercontent.google.com/download?id={id}'
     '&export=download&confirm=t'),
    ('drive', 'Google Drive 直链',
     'https://drive.google.com/uc?export=download&id={id}&confirm=t'),
    ('docs', 'Google Docs 直链',
     'https://docs.google.com/uc?export=download&id={id}&confirm=t'),
    ('gdown', 'gdown 官方文件协议', None),
]


def _npy_header_layout(path):
    """只读取 .npy 头部，返回形状、dtype、数据偏移和完整文件字节数。"""
    import math
    import numpy as np
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        elif version in ((2, 0), (3, 0)):
            # 固定 ASCII 键下，v2/v3 的长度字段和数据布局计算相同。
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise RuntimeError(f'不支持的 NumPy 格式版本: {version}')
        offset = int(f.tell())
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise RuntimeError('拒绝需要 pickle 的 object dtype')
    shape = tuple(int(v) for v in shape)
    payload_values = math.prod(shape)
    return {
        'shape': shape, 'dtype': dtype, 'offset': offset,
        'itemsize': int(dtype.itemsize),
        'file_bytes': int(offset + payload_values * dtype.itemsize),
    }


def _validate_bhn_file(path, name, expected_shape=None):
    """拒绝 HTML、物理截断/追加及发布布局 shape 不符的文件。"""
    if not os.path.exists(path) or os.path.getsize(path) < 2:
        raise RuntimeError('文件不存在或为空')
    with open(path, 'rb') as f:
        head = f.read(512)
    if b'<html' in head.lower() or b'<!doctype' in head.lower():
        raise RuntimeError('下载到 HTML 错误页而非数据')

    if name.endswith('.npy'):
        import numpy as np
        try:
            layout = _npy_header_layout(path)
        except Exception as e:
            raise RuntimeError(f'NumPy 头部校验失败: {e}') from e
        # 先检查物理大小，避免 shape 错误掩盖真正的截断或追加。
        actual_size = os.path.getsize(path)
        if actual_size != layout['file_bytes']:
            raise RuntimeError(
                '文件物理大小不符（可能截断或追加）: '
                f'{actual_size} != {layout["file_bytes"]}')
        try:
            arr = np.load(path, mmap_mode='r', allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f'NumPy 格式校验失败: {e}') from e
        if tuple(arr.shape) != layout['shape'] or arr.dtype != layout['dtype']:
            raise RuntimeError('NumPy 头部与只读 memmap 布局不一致')
        if (expected_shape is not None
                and layout['shape'] != tuple(expected_shape)):
            raise RuntimeError(
                '发布布局 shape 不符（文件物理完整，禁止据此自动重下）: '
                f'{layout["shape"]} != {tuple(expected_shape)}')
    elif name == 'data.pkl':
        # 不 pickle.load 网络文件, 仅检查二进制协议头与 STOP 结束符。
        with open(path, 'rb') as f:
            start = f.read(2)
            f.seek(-1, os.SEEK_END)
            stop = f.read(1)
        if (len(start) != 2 or start[:1] != b'\x80'
                or not 2 <= start[1] <= 5 or stop != b'.'):
            raise RuntimeError('pickle 协议头/结束符校验失败')


def diagnose_bitheteronet(data_dir='./data/bitheteronet', verbose=True):
    """无网络逐文件诊断；绝不反序列化 data.pkl。

    valid 仅覆盖文件物理完整性和 BHN_LAYOUT_PROFILE 发布布局。报告另行
    保留 README documented_shape，不能把布局通过解释为逐列语义已核验。
    """
    import math
    data_dir = os.path.abspath(data_dir)
    for old_name, new_name in BHN_ALIASES.items():
        old = os.path.join(data_dir, old_name)
        new = os.path.join(data_dir, new_name)
        if os.path.exists(old) and not os.path.exists(new):
            os.replace(old, new)
    report = {
        'data_dir': data_dir, 'network_access': False,
        'pickle_deserialized': False,
        'validation_scope': 'physical_integrity_and_release_array_layout',
        'layout_profile': BHN_LAYOUT_PROFILE,
        'column_semantics_fully_verified': False,
        'schema_references': dict(BHN_SCHEMA_REFERENCES),
        'files': {}, 'invalid': [], 'documentation_mismatch': [],
    }
    for name, meta in BHN_FILES.items():
        path = os.path.join(data_dir, name)
        documented_shape = meta.get('documented_shape', meta['shape'])
        item = {
            'path': path, 'exists': os.path.exists(path),
            'expected_shape': meta['shape'],
            'expected_layout_shape': meta['shape'],
            'documented_shape': documented_shape,
            'layout_note': meta.get('layout_note'),
            'semantic_status': meta.get('semantic_status'),
            'physical_complete': None,
            'layout_shape_consistent': None,
            'documented_shape_consistent': None,
            'valid': False, 'status': 'pending', 'error': None,
        }
        if item['exists']:
            item['actual_bytes'] = os.path.getsize(path)
        try:
            if name.endswith('.npy') and item['exists']:
                try:
                    layout = _npy_header_layout(path)
                    actual_bytes = item['actual_bytes']
                    item.update(
                        actual_shape=list(layout['shape']),
                        dtype=str(layout['dtype']),
                        header_data_offset=layout['offset'],
                        header_implied_file_bytes=layout['file_bytes'],
                        physical_complete=(
                            actual_bytes == layout['file_bytes']),
                        layout_shape_consistent=(
                            layout['shape'] == tuple(meta['shape'])),
                    )
                    if documented_shape is not None:
                        item['documented_shape_consistent'] = (
                            layout['shape'] == tuple(documented_shape))
                        item['documented_shape_implied_file_bytes'] = int(
                            layout['offset'] + math.prod(documented_shape)
                            * layout['itemsize'])
                except Exception:
                    # 统一校验器保留原始异常；已解析的头部字段仍保留。
                    pass
            _validate_bhn_file(path, name, meta['shape'])
            item['valid'] = True
            item['status'] = 'valid'
            if item['documented_shape_consistent'] is False:
                item['status'] = 'valid_layout_documented_shape_mismatch'
                report['documentation_mismatch'].append(name)
        except Exception as e:
            item['error'] = f'{type(e).__name__}: {e}'
            if (item['physical_complete'] is True
                    and item['layout_shape_consistent'] is False):
                item['status'] = 'shape_mismatch_physical_complete'
            elif item['physical_complete'] is False:
                item['status'] = 'physical_size_mismatch'
            else:
                item['status'] = 'invalid_or_missing'
            report['invalid'].append(name)
        report['files'][name] = item
        if verbose:
            print(f'\n[BHN-VALIDATE] {name}: '
                  f'{"OK" if item["valid"] else "INVALID"}; '
                  f'status={item["status"]}')
            print(f'  path={path}')
            print(f'  exists={item["exists"]}; '
                  f'actual_bytes={item.get("actual_bytes")}')
            print(f'  expected_layout_shape={meta["shape"]}; '
                  f'documented_shape={documented_shape}; '
                  f'actual_shape={item.get("actual_shape")}; '
                  f'dtype={item.get("dtype")}')
            if 'header_implied_file_bytes' in item:
                print('  header_implied_file_bytes='
                      f'{item["header_implied_file_bytes"]}; '
                      'documented_shape_implied_file_bytes='
                      f'{item.get("documented_shape_implied_file_bytes")}')
                print(f'  physical_complete={item["physical_complete"]}; '
                      'layout_shape_consistent='
                      f'{item["layout_shape_consistent"]}; '
                      'documented_shape_consistent='
                      f'{item["documented_shape_consistent"]}')
            if item['layout_note']:
                print(f'  layout_note={item["layout_note"]}')
            if item['error']:
                print(f'  reason={item["error"]}')
    if verbose:
        print('\n[BHN-VALIDATE] 无网络校验完成；data.pkl 未执行 pickle.load。')
        print('[BHN-VALIDATE] valid 不代表逐列语义已独立核验。')
        print(f'[BHN-VALIDATE] 文档形状不一致: '
              f'{report["documentation_mismatch"]}')
        print(f'[BHN-VALIDATE] 待处理文件: {report["invalid"]}')
    return report


def _bhn_missing_files(data_dir, verbose=False):
    """兼容官方旧文件名，返回缺失或校验失败文件；可保留失败详情。"""
    report = diagnose_bitheteronet(data_dir, verbose=verbose)
    return report['invalid']


def _file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def _gdown_download_file(file_id, dest):
    """gdown 单文件通道; 兼容不支持 timeout/retries 的旧版本。"""
    import inspect
    import gdown
    kwargs = {
        'url': f'https://drive.google.com/uc?id={file_id}',
        'output': dest, 'quiet': False, 'use_cookies': False,
        'resume': True, 'timeout': (20, 60), 'retries': 2,
    }
    supported = inspect.signature(gdown.download).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in supported}
    out = gdown.download(**kwargs)
    if out and os.path.abspath(out) != os.path.abspath(dest):
        os.replace(out, dest)
    if not os.path.exists(dest):
        raise RuntimeError('gdown 未生成目标文件')


def ensure_bitheteronet(data_dir='./data/bitheteronet', verbose=True):
    """下载作者官方 BitHeteroNet, 免账号、逐文件断点续传与通道切换。

    固定文件 ID 绕过文件夹列表页; UserContent -> Drive -> Docs -> gdown
    自动切换。通道的 .part 相互隔离, 不会拼接不同来源的未完成字节。
    所有必需文件通过格式/shape/大小校验后才判定成功。
    """
    os.makedirs(data_dir, exist_ok=True)
    report = diagnose_bitheteronet(data_dir, verbose=verbose)
    missing = report['invalid']
    if not missing:
        if verbose:
            print(f'[bitheteronet] 6 个官方文件布局均已校验, 跳过: {data_dir}')
            print('[bitheteronet] 文档形状差异保留警告；逐列语义未独立核验')
        return data_dir
    blocked = [
        name for name in missing
        if report['files'][name]['physical_complete'] is True
        and report['files'][name]['layout_shape_consistent'] is False]
    if blocked:
        # 完整但未知 shape 不是可续传的截断文件；重复下载同一 ID 无意义。
        print('[bitheteronet] 检测到物理完整但发布布局未知的文件, '
              f'不自动重下或删除: {blocked}')
        print('[bitheteronet] 请先核对版本/列语义；仅 --validate_only 诊断, '
              '不按本地实际 shape 自动放行')
        return False

    if verbose:
        print('[bitheteronet] 使用作者官方固定文件 ID, 跳过文件夹列表解析')
        print('[bitheteronet] 主通道 + 3 备用通道; 超时/格式错误自动切换')
    channels = list(BHN_CHANNELS)
    used, errors = {}, []
    for name in BHN_FILES:
        if name not in missing:
            continue
        meta = BHN_FILES[name]
        dest = os.path.join(data_dir, name)
        succeeded = False
        for index in range(len(channels)):
            key, label, template = channels[index]
            staging = dest + f'.{key}.download'
            candidate_complete = False
            try:
                if verbose:
                    print(f'[bitheteronet] {name} <- {label}')
                if os.path.exists(staging):
                    candidate_complete = True   # 上次已下载但未来得及校验
                elif os.path.exists(staging + '.part'):
                    # 进程可能在最后一个字节写入后、原子重命名前退出。
                    # 完整 .part 直接接管，避免按文件末尾 Range 请求收到 416。
                    try:
                        _validate_bhn_file(staging + '.part', name, meta['shape'])
                    except Exception:
                        pass                # 真正的未完成断点，下面继续续传
                    else:
                        os.replace(staging + '.part', staging)
                        candidate_complete = True
                if not candidate_complete and key == 'gdown':
                    _gdown_download_file(meta['id'], staging)
                    candidate_complete = True
                elif not candidate_complete:
                    download_with_resume(
                        template.format(id=meta['id']), staging,
                        max_retry=1, timeout=25)
                    candidate_complete = True
                _validate_bhn_file(staging, name, meta['shape'])
                os.replace(staging, dest)
            except Exception as e:
                # 完整候选若校验失败必须删除; 网络中断留下的 .part 则保留。
                if candidate_complete and os.path.exists(staging):
                    os.remove(staging)
                errors.append(f'{name} [{label}]: {str(e)[:140]}')
                print(f'[bitheteronet] 通道失败, 自动切换: {str(e)[:140]}')
                continue
            used[name] = label
            channels.insert(0, channels.pop(index))  # 后续文件优先成功通道
            succeeded = True
            if verbose:
                print(f'[bitheteronet] 完成并校验: {name}')
            break
        if not succeeded:
            break   # 所有通道均失败, 保留断点并进入手动提示

    final_report = diagnose_bitheteronet(data_dir, verbose=False)
    missing = final_report['invalid']
    if missing:
        print(f'[bitheteronet] 自动下载未完成, 待处理文件: {missing}')
        for err in errors[-8:]:
            print(f'  {err}')
        print('[bitheteronet] 手动步骤 (免账号):')
        print(f'  1. 浏览器打开 {BHN_DRIVE_URL}')
        print(f'  2. 下载全部 6 个文件到 {data_dir}/')
        print('  3. 重跑本脚本; 已校验文件跳过, 未完成文件从 .part 续传')
        return False

    manifest = {
        'dataset': 'BitHeteroNet',
        'official_repository': f'https://github.com/{BHN_REPO}',
        'official_folder': BHN_DRIVE_URL,
        'verified_on': '2026-10-26',
        'layout_profile': BHN_LAYOUT_PROFILE,
        'validation_scope': final_report['validation_scope'],
        'column_semantics_fully_verified': False,
        'documentation_mismatch': final_report['documentation_mismatch'],
        'schema_references': dict(BHN_SCHEMA_REFERENCES),
        'note': '同一作者官方数据的多个下载通道, 非独立第三方镜像; '
                '布局通过不等于逐列语义已独立核验',
        'files': {},
    }
    for name, meta in BHN_FILES.items():
        path = os.path.join(data_dir, name)
        manifest['files'][name] = {
            'file_id': meta['id'], 'bytes': os.path.getsize(path),
            'sha256': _file_sha256(path),
            'expected_layout_shape': meta['shape'],
            'documented_shape': meta.get('documented_shape', meta['shape']),
            'channel': used.get(name, 'local/verified'),
        }
    manifest_path = os.path.join(data_dir, 'download_manifest.json')
    with open(manifest_path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(manifest_path + '.tmp', manifest_path)
    if verbose:
        print(f'[bitheteronet] 全部 6 个文件校验通过: {data_dir}')
        print(f'[bitheteronet] 来源与 SHA-256 清单: {manifest_path}')
    return data_dir


# ------------------------------------------------------------------
# 统一入口
# ------------------------------------------------------------------
REGISTRY = {
    'elliptic': ensure_elliptic,
    'ethereum': ensure_ethereum,
    'dgraphfin': ensure_dgraphfin,
    'dexposure': ensure_dexposure,
    'bitheteronet': ensure_bitheteronet,
}


def ensure_dataset(name, data_root='./data', **kw):
    fn = REGISTRY[name]
    if name == 'dexposure':
        return fn(os.path.join(data_root, name), **kw)
    return fn(os.path.join(data_root, name))


def main():
    ap = argparse.ArgumentParser(description='TASGL 全部数据集免账号下载(断点续传)')
    ap.add_argument('--dataset', default='all',
                    choices=['all'] + list(REGISTRY))
    ap.add_argument('--data_root', default='./data')
    ap.add_argument('--full', action='store_true',
                    help='DeXposure 下载 1GB 全量快照 (默认 75MB 快速快照)')
    ap.add_argument('--validate_only', action='store_true',
                    help='仅无网络详细校验 BitHeteroNet，不发起任何下载')
    ap.add_argument('--validation_report', default=None,
                    help='可选：将 BitHeteroNet 校验报告写入指定 JSON')
    args = ap.parse_args()

    if args.validate_only:
        data_dir = os.path.join(args.data_root, 'bitheteronet')
        report = diagnose_bitheteronet(data_dir, verbose=True)
        if args.validation_report:
            report_path = os.path.abspath(args.validation_report)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            os.replace(report_path + '.tmp', report_path)
            print(f'[BHN-VALIDATE] JSON 报告: {report_path}')
        raise SystemExit(2 if report['invalid'] else 0)

    names = list(REGISTRY) if args.dataset == 'all' else [args.dataset]
    failed = []
    for name in names:
        print(f'\n========== 数据集: {name} ==========')
        try:
            r = ensure_dataset(name, args.data_root, full=args.full) \
                if name == 'dexposure' else ensure_dataset(name, args.data_root)
            if r is False:
                failed.append(name)
        except Exception as e:
            print(f'[{name}] 失败: {str(e)[:300]}')
            failed.append(name)
    print('\n========== 下载汇总 ==========')
    for name in names:
        print(f'  {name}: {"需手动处理" if name in failed else "OK"}')
    if failed:
        print('提示: 失败项不影响其他数据集与后续 Elliptic 主实验。')


if __name__ == '__main__':
    main()
