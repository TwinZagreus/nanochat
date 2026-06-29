"""
沙盒执行工具: 用于安全运行LLM生成的Python代码。
改编自OpenAI HumanEval代码:
https://github.com/openai/human-eval/blob/master/human_eval/execution.py

Sandboxed execution utilities for running Python code that comes out of an LLM.
Adapted from OpenAI HumanEval code:
https://github.com/openai/human-eval/blob/master/human_eval/execution.py

已覆盖的安全措施 / What is covered:
- 每次执行在独立进程中运行(挂起或崩溃时可被终止)
  Each execution runs in its own process (can be killed if it hangs or crashes)
- 通过超时限制防止无限循环
  Execution is limited by a timeout to stop infinite loops
- 默认强制执行内存限制(256MB)
  Memory limits are enforced by default (256MB)
- 捕获并返回stdout和stderr
  stdout and stderr are captured and returned
- 代码在临时目录中运行,运行后自动删除
  Code runs in a temporary directory that is deleted afterwards
- 禁用危险函数(os.system, os.kill, shutil.rmtree, subprocess.Popen等)
  Dangerous functions are disabled (examples: os.system, os.kill, shutil.rmtree, subprocess.Popen)

未覆盖的限制 / What is not covered:
- 不是真正的安全沙盒
  Not a true security sandbox
- 网络访问未被阻止(例如可以打开socket)
  Network access is not blocked (e.g. sockets could be opened)
- Python动态特性(如ctypes)可能绕过限制
  Python's dynamic features (e.g. ctypes) could bypass restrictions
- 没有内核级隔离(无seccomp, 无容器, 无虚拟化)
  No kernel-level isolation (no seccomp, no containers, no virtualization)

总体而言,此沙盒适用于评估生成代码,可防止意外破坏行为,但无法防御恶意对抗性代码。
Overall this sandbox is good for evaluation of generated code and protects against
accidental destructive behavior, but it is not safe against malicious adversarial code.
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from dataclasses import dataclass
from typing import Optional

# -----------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """沙盒中执行Python代码的结果。 / Result of executing Python code in a sandbox."""
    success: bool         # 执行是否成功 / Whether execution succeeded
    stdout: str           # 捕获的标准输出 / Captured stdout
    stderr: str           # 捕获的标准错误 / Captured stderr
    error: Optional[str] = None        # 错误信息(如有) / Error message if any
    timeout: bool = False              # 是否超时 / Whether execution timed out
    memory_exceeded: bool = False      # 是否超出内存限制 / Whether memory limit was exceeded

    def __repr__(self):
        parts = []
        parts.append(f"ExecutionResult(success={self.success}")
        if self.timeout:
            parts.append(", timeout=True")
        if self.memory_exceeded:
            parts.append(", memory_exceeded=True")
        if self.error:
            parts.append(f", error={self.error!r}")
        if self.stdout:
            parts.append(f", stdout={self.stdout!r}")
        if self.stderr:
            parts.append(f", stderr={self.stderr!r}")
        parts.append(")")
        return "".join(parts)


@contextlib.contextmanager
def time_limit(seconds: float):
    """
    上下文管理器: 在指定秒数后通过SIGALRM触发超时异常。
    Context manager that raises a TimeoutException after the given number of seconds via SIGALRM.
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
        # 超时! / Timed out!

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)  # 清除定时器 / Clear the timer


@contextlib.contextmanager
def capture_io():
    """捕获stdout和stderr，并禁用stdin。 / Capture stdout and stderr, and disable stdin."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    stdin_block = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stdout_capture):
        with contextlib.redirect_stderr(stderr_capture):
            with redirect_stdin(stdin_block):
                yield stdout_capture, stderr_capture


@contextlib.contextmanager
def create_tempdir():
    """
    上下文管理器: 创建临时目录并切换工作目录到其中，退出时自动清理。
    Context manager that creates a temporary directory, chdirs into it, and auto-cleans on exit.
    """
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    """超时异常: 当代码执行超过时间限制时抛出。 / Raised when code execution exceeds the time limit."""
    pass


class WriteOnlyStringIO(io.StringIO):
    """只写不读的StringIO: 读取时抛出异常,用于阻止stdin。 / StringIO that throws an exception when it's read from, used to block stdin."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    """重定向stdin的上下文管理器(继承自标准库_RedirectStream)。 / Context manager for redirecting stdin (inherits from stdlib _RedirectStream)."""
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    """
    上下文管理器: 切换到指定目录，退出时恢复原工作目录。
    Context manager that changes to the given directory and restores the original cwd on exit.
    """
    if root == ".":  # 当前目录无需切换 / No need to change if already in target
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)  # 恢复原目录 / Restore original cwd


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    禁用各种破坏性函数,防止生成代码干扰测试环境(如fork炸弹、杀进程、删文件等)。
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    警告: 这不是安全沙盒。不可信代码(包括模型生成代码)不应在沙盒外盲目执行。
    请参阅Codex论文了解OpenAI代码沙盒的更多信息,并谨慎使用。
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if platform.uname().system != "Darwin":
        # 这些资源限制调用在macOS(Darwin)上似乎会失败，跳过 / These resource limit calls seem to fail on macOS (Darwin), skip?
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore
    # 禁用subprocess.Popen以防止生成代码启动外部进程 / Disable subprocess.Popen

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _unsafe_execute(code: str, timeout: float, maximum_memory_bytes: Optional[int], result_dict):
    """
    在子进程中执行代码并施加安全保护。结果写入共享字典result_dict。
    Execute code in a subprocess with safety guards. Results are written to result_dict.
    """
    with create_tempdir():

        # 清理临时目录时需要这些系统调用 / These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        # 禁用可能造成破坏性更改的功能 / Disable functionalities that can make destructive changes to the test.
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)

        # 默认为失败状态 / Default to failure
        result_dict.update({
            "success": False,
            "stdout": "",
            "stderr": "",
            "timeout": False,
            "memory_exceeded": False,
            "error": None,
        })

        try:
            exec_globals = {}
            with capture_io() as (stdout_capture, stderr_capture):
                with time_limit(timeout):
                    # 警告: 此程序执行不可信的模型生成代码。虽然模型生成代码不太可能
                    # 做出恶意行为,但可能因模型能力或对齐不足而产生破坏性操作。
                    # 强烈建议用户对此评估套件进行沙盒隔离,防止其对主机或网络造成破坏。
                    # 有关OpenAI如何沙盒化其代码的更多信息,请参阅随附论文。
                    # 阅读此免责声明并采取适当预防措施后,可以取消下行注释并自担风险执行:
                    # WARNING
                    # This program exists to execute untrusted model-generated code. Although
                    # it is highly unlikely that model-generated code will do something overtly
                    # malicious in response to this test suite, model-generated code may act
                    # destructively due to a lack of model capability or alignment.
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # does not perform destructive actions on their host or network. For more
                    # information on how OpenAI sandboxes its code, see the accompanying paper.
                    # Once you have read this disclaimer and taken appropriate precautions,
                    # uncomment the following line and proceed at your own risk:
                    exec(code, exec_globals)

            result_dict.update({
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            })

        except TimeoutException:
            result_dict.update({
                "timeout": True,
                "error": "Execution timed out",  # 执行超时 / Execution timed out
            })

        except MemoryError as e:
            result_dict.update({
                "memory_exceeded": True,
                "error": f"Memory limit exceeded: {e}",  # 内存超限 / Memory limit exceeded
            })

        except BaseException as e:
            result_dict.update({
                "error": f"{type(e).__name__}: {e}",
            })

        # 恢复系统调用以用于清理临时目录 / Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink


def execute_code(
    code: str,
    timeout: float = 5.0, # 默认5秒 / 5 seconds default
    maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024, # 默认256MB / 256MB default
) -> ExecutionResult:
    """
    在沙盒环境中执行Python代码。 / Execute Python code in a sandboxed environment.

    Args:
        code: 待执行的Python代码字符串 / Python code to execute as a string
        timeout: 最大执行时间(秒), 默认5.0 / Maximum execution time in seconds (default: 5.0)
        maximum_memory_bytes: 内存限制(字节), 默认256MB, None表示不限制 / Memory limit in bytes (default: 256MB, None to disable)

    Returns:
        包含成功状态、stdout/stderr和错误信息的ExecutionResult
        ExecutionResult with success status, stdout/stderr, and error information

    Example:
        >>> result = execute_code("print('hello world')")
        >>> result.success
        True
        >>> result.stdout
        'hello world\\n'
    """

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    p = multiprocessing.Process(
        target=_unsafe_execute,
        args=(code, timeout, maximum_memory_bytes, result_dict)
    )
    p.start()
    p.join(timeout=timeout + 1)  # 额外+1秒作为加入缓冲 / Extra +1s as join buffer

    if p.is_alive():  # 进程仍在运行, 强制终止 / Process still alive, force kill
        p.kill()
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution timed out (process killed)",  # 执行超时(进程被终止) / Execution timed out (process killed)
            timeout=True,
            memory_exceeded=False,
        )

    if not result_dict:  # 子进程崩溃,无结果返回 / Child process crashed, no result
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution failed (no result returned)",  # 执行失败(无结果返回) / Execution failed (no result returned)
            timeout=True,
            memory_exceeded=False,
        )

    return ExecutionResult(
        success=result_dict["success"],
        stdout=result_dict["stdout"],
        stderr=result_dict["stderr"],
        error=result_dict["error"],
        timeout=result_dict["timeout"],
        memory_exceeded=result_dict["memory_exceeded"],
    )

