import os

def is_same_var(var1, var2):
    return var1["name"] == var2["name"] and var1["value"] == var2["value"]

def metrics(pred, label):
    assert pred.shape == label.shape
    TP = FP = TN = FN = 0
    for p, l in zip(pred, label):
        if l == 1 and p == 1:
            TP += 1
        elif l == 1 and p == 0:
            FN += 1
        elif l == 0 and p == 1:
            FP += 1
        elif l == 0 and p == 0:
            TN += 1
        else:
            print("error")
    print("     TP: %d, FP: %d, TN: %d, FN: %d" % (TP, FP, TN, FN))
    print("     precision: %f, recall: %f" % (TP/(TP+FP), TP/(TP+FN)))

def list_to_str(test_list):
    test_str = " "
    test_str = test_str + " ".join([str(x) for x in test_list])
    return test_str

def last_line_over(line_b):
    line_str = line_b.decode("utf-8")
    return line_str.startswith("  selection") or line_str.startswith("  Avg. Gap") or line_str.startswith("Gap") 

def is_file_over(filename):
    """
    get last line of a file
    :param filename: file name
    :return: last line starts with "   selection time"
    """
    try:
        filesize = os.path.getsize(filename)
        if filesize == 0:
            return False
        else:
            with open(filename, 'rb') as fp: # to use seek from end, must use mode 'rb'
                offset = -8                 # initialize offset
                while -offset < filesize:   # offset cannot exceed file size
                    fp.seek(offset, 2)      # read # offset chars from eof(represent by number '2')
                    lines = fp.readlines()  # read from fp to eof
                    if len(lines) >= 2:     # if contains at least 2 lines
                        return last_line_over(lines[-1])    # then last line is totally included
                    else:
                        offset *= 2         # enlarge offset
                fp.seek(0)
                lines = fp.readlines()
                return last_line_over(lines[-1])
    except FileNotFoundError:
        print(' not found!')
        print(filename)
        # input()
        return False
