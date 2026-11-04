"""P1-7：匿名 staging、数据血缘、哈希和只读原始结果审计。"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from p0_core import atomic_json, file_sha256, read_json, sha256_bytes


# Representative legacy Windows interpreter signature used by the
# anonymising staging pass.  The development machine's real user name is
# deliberately not published; this generic stand-in keeps the redaction
# rule and its unit tests reproducible on any machine.
FIXED_WINDOWS_INTERPRETER = ('C:' + '\\Users\\example\\anaconda3\\python.exe')
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.csv', '.bat', '.sh',
                 '.tex', '.bib', '.yml', '.yaml'}


def _patterns():
    return {
        'private_key': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
        'windows_user_path': re.compile(r'C:\\Users\\[^\\\s]+', re.I),
        'posix_home_path': re.compile('/' + r'(?:home|Users)/[^/\s]+'),
        'agent_upload_path': re.compile('/' + r'mnt/agents/upload'),
        'assigned_secret': re.compile(
            r'(?i)(?:api[_-]?key|appsecret|password|access[_-]?token)\s*[:=]\s*["\'][^"\']+["\']'),
    }


def sensitive_findings(text):
    findings = []
    for name, pattern in _patterns().items():
        matches = sorted(set(m.group(0) for m in pattern.finditer(text)))
        if matches:
            findings.append({'type': name, 'count': len(matches),
                             'examples': [m[:120] for m in matches[:3]]})
    return findings


def _safe_relative(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'发布路径必须为根目录内相对路径: {value}')
    source = (root / path).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f'发布文件越过 source_root: {value}') from exc
    if not source.is_file() or source.is_symlink():
        raise ValueError(f'发布输入不是普通文件或含符号链接: {value}')
    return path, source


def stage_release(source_root, output_dir, includes, lineage,
                  raw_result_files=(), anonymize_launchers=True):
    """不修改源文件；原始结果不脱敏、不改写，只复制并设只读。"""
    root = Path(source_root).resolve()
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('匿名发布目录非空；禁止覆盖，请使用新目录')
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for value in includes:
        rel, src = _safe_relative(root, value)
        entries[str(rel)] = (src, False)
    for value in raw_result_files:
        rel, src = _safe_relative(root, value)
        if src.suffix.lower() in {'.pkl', '.pickle'}:
            raise ValueError('网络 pickle 不进入发布包；只保留静态校验报告')
        entries[str(rel)] = (src, True)
    manifest_files = []
    binary_privacy_review = []
    for rel, (src, is_raw) in sorted(entries.items()):
        original_hash = file_sha256(src)
        transforms = []
        if is_raw or src.suffix.lower() not in TEXT_SUFFIXES:
            payload = src.read_bytes()
            if is_raw and src.suffix.lower() in TEXT_SUFFIXES:
                # 原始文本结果绝不脱敏；若含敏感内容，阻断发布而不是改写。
                findings = sensitive_findings(payload.decode('utf-8'))
                if findings:
                    raise ValueError(f'原始结果含敏感内容，禁止发布 {rel}: {findings}')
            elif src.suffix.lower() not in TEXT_SUFFIXES:
                binary_privacy_review.append(rel)
        else:
            # 字节读取后解码，避免 Windows 通用换行把 CRLF 规范化为 LF；
            # 除显式记录的匿名替换外，staging 副本必须保留原始字节与换行。
            text = src.read_bytes().decode('utf-8')
            if anonymize_launchers:
                # 同时处理 BAT/文本中的单反斜杠和 Python 源中的转义写法；
                # 仅修改 staging 副本，且记录每一次语义变换。
                variants = {FIXED_WINDOWS_INTERPRETER,
                            FIXED_WINDOWS_INTERPRETER.replace('\\', '\\\\')}
                replaced = False
                for variant in sorted(variants, key=len, reverse=True):
                    if variant in text:
                        text = text.replace(variant, 'python.exe')
                        replaced = True
                if replaced:
                    transforms.append('fixed_windows_interpreter_replaced_with_PATH_python')
            findings = sensitive_findings(text)
            if findings:
                raise ValueError(f'匿名审计阻止发布 {rel}: {findings}')
            payload = text.encode('utf-8')
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(payload)
        if is_raw:
            dst.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        manifest_files.append({
            'path': rel, 'bytes': len(payload), 'original_sha256': original_hash,
            'staged_sha256': sha256_bytes(payload), 'raw_result': is_raw,
            'read_only_in_staging': is_raw, 'transforms': transforms,
        })
    manifest = {
        'protocol': 'p1_anonymous_release_v1',
        'anonymous_ready': not binary_privacy_review,
        'anonymity_status': ('automated_text_scan_passed'
                             if not binary_privacy_review else
                             'pending_external_binary_privacy_review'),
        'binary_privacy_review_required': sorted(binary_privacy_review),
        'source_files_unmodified': True, 'raw_results_unmodified': True,
        'raw_results_read_only': True,
        'network_pickle_deserialized': False,
        'lineage': lineage,
        'lineage_boundary': ('Sources are declared from the implemented download paths. '
                             'Multiple access channels to one author dataset are not independent academic mirrors.'),
        'files': manifest_files,
    }
    atomic_json(out / 'release_manifest.json', manifest)
    return manifest


def verify_release(output_dir):
    out = Path(output_dir)
    manifest = read_json(out / 'release_manifest.json')
    errors = []
    listed = {'release_manifest.json'}
    for item in manifest['files']:
        path = out / item['path']
        listed.add(item['path'])
        if not path.is_file():
            errors.append(f'missing:{item["path"]}')
            continue
        if file_sha256(path) != item['staged_sha256']:
            errors.append(f'hash:{item["path"]}')
        if item['raw_result']:
            mode = path.stat().st_mode
            if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                errors.append(f'raw_not_read_only:{item["path"]}')
        if path.suffix.lower() in TEXT_SUFFIXES:
            findings = sensitive_findings(path.read_text(encoding='utf-8'))
            if findings:
                errors.append(f'sensitive:{item["path"]}:{findings}')
    extras = {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()} - listed
    if extras:
        errors.append(f'unlisted_files:{sorted(extras)}')
    return {'valid': not errors, 'errors': errors,
            'n_files': len(manifest['files']),
            'anonymous_ready': bool(manifest.get('anonymous_ready')),
            'anonymity_status': manifest.get('anonymity_status'),
            'binary_privacy_review_required':
                manifest.get('binary_privacy_review_required', []),
            'network_pickle_deserialized': False}
