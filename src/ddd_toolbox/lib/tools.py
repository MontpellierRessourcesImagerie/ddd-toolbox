class RenameListElements(object):


    def __init__(self, names):
        self.names = names
        self.result = None
        self.subString = "_"
        self.newSubString = "-"
        self.isFlip = False


    def run(self):
        self.result = []
        for name in self.names:
            newName = name
            if self.isFlip:
                parts = name.split(self.subString)
                newName = parts[-1] + self.subString + parts[-2]
            newName = newName.replace(self.subString, self.newSubString)
            print(newName, self.subString, self.newSubString)
            self.result.append(newName)
        print(self.result)
