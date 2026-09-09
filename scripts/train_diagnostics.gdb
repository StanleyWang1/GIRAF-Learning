# Run with: uv run --frozen gdb -nx -q -batch -x scripts/train_diagnostics.gdb \
#   --args python -m giraf.learning.train_cli <training arguments>
# Redirect/tee stdout and stderr to the run's train.log.
set pagination off
set confirm off
set print thread-events off
set debuginfod enabled off
set disable-randomization off

python
import datetime
import json
from pathlib import Path
import signal
import gdb

diagnostic_exit_code = 1
diagnostic_signal = None

def diagnostic_on_exit(event):
    global diagnostic_exit_code
    diagnostic_exit_code = getattr(event, 'exit_code', 1)

def diagnostic_on_stop(event):
    global diagnostic_signal, diagnostic_exit_code
    if isinstance(event, gdb.SignalEvent):
        diagnostic_signal = event.stop_signal
        diagnostic_exit_code = 128 + getattr(signal, event.stop_signal, 0)

def diagnostic_read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return str(error)

def diagnostic_cpu(cpu):
    base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    return {
        'logical_cpu': cpu,
        'raw_core_id': diagnostic_read(base / 'core_id'),
        'socket': diagnostic_read(base / 'physical_package_id'),
        'smt_siblings': diagnostic_read(base / 'thread_siblings_list'),
    }

gdb.events.exited.connect(diagnostic_on_exit)
gdb.events.stop.connect(diagnostic_on_stop)
print('[DIAGNOSTICS] started_at=' + datetime.datetime.now().astimezone().isoformat())
print('[DIAGNOSTICS] bios=' + diagnostic_read('/sys/class/dmi/id/bios_version'))
print('[DIAGNOSTICS] online_cpus=' + diagnostic_read('/sys/devices/system/cpu/online'))
print('[DIAGNOSTICS] CPU IDs below are Linux logical CPU IDs; raw core IDs may differ from lscpu numbering.')
end

run

python
print('[DIAGNOSTICS] stopped_at=' + datetime.datetime.now().astimezone().isoformat())
thread = gdb.selected_thread()
if diagnostic_signal is not None and thread is not None:
    pid = gdb.selected_inferior().pid
    tid = thread.ptid[1]
    report = {'signal': diagnostic_signal, 'pid': pid, 'tid': tid}
    path = Path(f'/proc/{pid}/task/{tid}')
    try:
        # stat field 39 is processor; the suffix after comm starts at field 3.
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        report.update(diagnostic_cpu(int(fields[36])))
        for line in (path / 'status').read_text().splitlines():
            if line.startswith('Cpus_allowed_list:'):
                report['allowed_cpus'] = line.split(':', 1)[1].strip()
    except (OSError, ValueError, IndexError) as error:
        report['cpu_lookup_error'] = str(error)
    print('[DIAGNOSTICS] fault=' + json.dumps(report, sort_keys=True))
    print('[DIAGNOSTICS] This CPU was running the stopped thread; it is not proof that this core caused the corruption.')
    temperatures = {}
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        if diagnostic_read(hwmon / 'name') != 'coretemp':
            continue
        for label in hwmon.glob('temp*_label'):
            reading = diagnostic_read(label.with_name(label.name.replace('_label', '_input')))
            temperatures[diagnostic_read(label)] = reading
    print('[DIAGNOSTICS] temperature_snapshot_millidegrees_c=' + json.dumps(temperatures, sort_keys=True))
    print('[DIAGNOSTICS] Temperatures are a post-stop snapshot, not a continuous history.')
    for command in ('bt 40', 'info registers', 'x/12i $pc', 'thread apply all bt 12', 'info sharedlibrary'):
        print('[DIAGNOSTICS] ' + command)
        try:
            gdb.execute(command)
        except gdb.error as error:
            print('[DIAGNOSTICS] ' + str(error))
print('[DIAGNOSTICS] exit_status=' + str(diagnostic_exit_code))
gdb.execute('quit ' + str(diagnostic_exit_code))
end
