import subprocess
import re
from distutils.util import strtobool

STATEMENT_COV = "Statement"
BRANCH_COV = "Branch"
CONDITION_COV = "Condition"

cov_re = re.compile(r'(\d+\.\d+)% \((\d+) / (\d+)\)')

def run_pcov(target, *args):
	cmd = ["python3", "pcov.py", "-t", target] + list(args)
	ret = subprocess.run(cmd, capture_output=True, text=True)
	return ret.stdout

def run_pcov_verbose(target, *args):
	cmd = ["python3", "pcov.py", "-v", "-t", target] + list(args)
	ret = subprocess.run(cmd, capture_output=True, text=True)
	return ret.stdout

def get_coverage(cov_type, output):
	_line = list(filter(lambda x: x.startswith(cov_type), output.split("\n")))[0]
	match = cov_re.search(_line)
	if match:
		return float(match.group(1)), int(match.group(2)), int(match.group(3))
	else:
		return 0, 0, 0

def get_verbose_output(output):
	_lines = list(filter(lambda x: x.startswith("Line"), output.split("\n")))
	return _lines

def process_verbose_output(lines):
	return [(int(x.split(":")[0].split(" ")[1]), bool(strtobool(x.split("==>")[1].strip()))) for x in lines]
