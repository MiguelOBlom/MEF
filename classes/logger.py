class Logger():
    def __init__(self):
        pass
    
    @staticmethod
    def note(message):
        print(f"\033[45mNote: {message}\033[0m")

    @staticmethod
    def warn(message):
        print(f"\033[43mWarning: {message}\033[0m")

    @staticmethod
    def fail(message):
        print(f"\033[91mError: {message}\033[0m")
        exit()
    
    @staticmethod
    def ok(message):
        print(f"\033[102mOK: {message}\033[0m")
