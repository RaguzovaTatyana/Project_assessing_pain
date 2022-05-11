import zipfile

def unzip_folder(folder_path, target_folder):
    with zipfile.ZipFile(folder_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

def open_file(name):
    f = open(name + '.txt', "r")
    scorestr = f.read()
    f.close()
    return scorestr

def get_frame(scorestr):
    return float(scorestr[0:scorestr.find('e')]) * (10 ** int(scorestr[scorestr.find('+') + 1:]))