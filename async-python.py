import asyncio

async def run(cmd):

    print(f"running: {cmd}")

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await proc.communicate()

    print(f'[{cmd!r} exited with {proc.returncode}]')
    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')

async def main():
    await asyncio.gather(
            *[run("sleep 10") for _ in range(10)]
            )

asyncio.run(main())
