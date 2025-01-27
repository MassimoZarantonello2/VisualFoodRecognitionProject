import time
class LogClass():
    def __init__(self, log_file):
        self.log_file = log_file
        self.log = open(self.log_file, 'a')

    def write(self, text, data=None):
        time_stamp = time.strftime('%H:%M:%S', time.localtime())
        if text == '\n':
            self.log.write('\n')
            return
        # controlla se data è una lista
        if isinstance(data, list):
            data = ', '.join([str(x) for x in data])
        # controlla se data è un dizionario
        if isinstance(data, dict):
            data = ', '.join([f'{key}: {value}' for key, value in data.items()])
        if data is None:
            self.log.write(f'{time_stamp}: {text}\n')
        else:
            self.log.write(f'{time_stamp}: {text} {data}\n')
        
    def close(self):
        self.log.close()