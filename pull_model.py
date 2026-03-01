from datetime import datetime
try:
    from datetime import UTC
except ImportError:
    # Python < 3.11 compatibility
    import datetime as dt
    UTC = dt.timezone.utc
from pathlib import Path
import subprocess
import sys
import traceback

PROJECT_DIR = Path(__file__).parent.resolve()
LOG_DIR = PROJECT_DIR / 'logs'

def _write_log(log_path: Path, command: list[str], returncode: int | None, stdout: str, stderr: str) -> None:
    with log_path.open('w', encoding='utf-8') as f:
        f.write(f'timestamp_utc: {datetime.now(UTC).isoformat()}\n')
        f.write(f'project_dir: {PROJECT_DIR}\n')
        f.write(f'return_code: {returncode}\n')
        f.write(f'command: {" ".join(command)}\n')
        f.write('\n===== STDOUT =====\n')
        f.write(stdout or '(empty)\n')
        f.write('\n===== STDERR =====\n')
        f.write(stderr or '(empty)\n')

def _run_export(command: list[str], label: str, log_path: Path) -> int:
    print(f'Exporting {label}...')
    result = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True, check=False)
    _write_log(log_path, command, result.returncode, result.stdout, result.stderr)
    if result.returncode != 0:
        print(f'{label} export failed.')
        print(result.stderr)
    else:
        print(f'{label} export complete.')
    return result.returncode

def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'openvino_export_{datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")}.log'
    python_bin = f'{PROJECT_DIR}/env/bin/python3'

    # 1. Llama 3.2 3B Instruct (INT4) - replaces TinyLlama
    command_llm = [
        python_bin,
        '-m', 'optimum.commands.optimum_cli',
        'export', 'openvino',
        '-m', 'meta-llama/Llama-3.2-3B-Instruct',
        '--task', 'text-generation-with-past',
        '--weight-format', 'int4',
        '--sym',
        str(PROJECT_DIR / 'models' / 'Llama-3.2-3B-Instruct-INT4'),
    ]

    # 2. bge-small-en-v1.5 embedding model - replaces all-MiniLM-L6-v2
    command_emb = [
        python_bin,
        '-m', 'optimum.commands.optimum_cli',
        'export', 'openvino',
        '-m', 'BAAI/bge-small-en-v1.5',
        '--task', 'feature-extraction',
        str(PROJECT_DIR / 'models' / 'bge-small-en-v1.5'),
    ]

    print('Starting OpenVINO model export...')
    print(f'Log file: {log_path}')

    ret = _run_export(command_llm, 'Llama-3.2-3B-Instruct INT4', log_path)
    if ret != 0:
        return ret

    ret = _run_export(command_emb, 'bge-small-en-v1.5', log_path)
    if ret != 0:
        return ret

    print('All exports finished successfully!')
    return 0

if __name__ == '__main__':
    sys.exit(main())
