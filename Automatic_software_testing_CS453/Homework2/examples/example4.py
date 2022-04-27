import sys

def count_gt(threshold, nums):
    gt = [int(x) for x in nums if int(x) > threshold]
    return len(gt)

s = "Great!" if count_gt(int(sys.argv[1]), sys.argv[2:]) > 0 else "Nah.."
print(s)