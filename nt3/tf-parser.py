#!/usr/bin/python3

import sys,re,glob

class LogParser:
    def __init__(self):
        self.BatchEx = re.compile("Rank \d+ - new batch, time since last batch: (\d+\.\d+)")
        self.CkptLocalEx = re.compile("Rank \d+ - checkpoint at epoch \d+, local ckpt duration: (\d+\.\d+)")
        self.CkptPrepEx = re.compile("Rank \d+ - checkpoint at epoch \d+, prepare duration: (\d+\.\d+)")
        self.avg = self.prep = self.local = self.after = 0.0
        self.count = self.acount = self.ckpt = 0

    def parse(self, file_name):
        self.start = 0
        afterCkpt = 1
        print("Parsing file %s" % file_name)
        for line in open(file_name,'r').readlines():
            e = self.CkptPrepEx.match(line)
            if e != None and len(e.groups()) == 1:
                self.prep += float(e.group(1))
            e = self.CkptLocalEx.match(line)
            if e != None and len(e.groups()) == 1:
                self.local += float(e.group(1))
                self.ckpt += 1
                afterCkpt = 0
            e = self.BatchEx.match(line)
            if e != None and len(e.groups()) == 1:
                if self.start > 4 and float(e.group(1)) < 20.0:
                    if self.start % 15 == 5 or self.start % 15 == 6:
                        #print("%.3f" % float(e.group(1)))
                        self.after += float(e.group(1))
                        self.acount += 1
                    else:
                        self.avg += float(e.group(1))
                        self.count += 1
                self.start += 1

    def __str__(self):
        print("Count/Ckpt: ", (self.count, self.ckpt))
        if self.ckpt > 0:
            print("CkptPrep/CkptLocal/CkptDelay: %.3f\t%.3f\t%.3f" % (self.prep / self.ckpt, self.local / self.ckpt, self.after / self.acount - self.avg / self.count))
        return "Avg/ckpt step: %.3f %.3f" % (self.avg / self.count, self.after / self.acount)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage %s <log_dir>" % sys.argv[0])

    lp = LogParser()
    f_no = 0
    for f in glob.glob(sys.argv[1] + "/*.log"):
        lp.parse(f)
        f_no += 1
    print("Statistics: %s" % lp)
    print("Total runtime: %.3f" % ((lp.avg + lp.after) / ((lp.count + lp.acount) * f_no)))
