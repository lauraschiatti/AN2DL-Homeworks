import os

i = 0


cwd = os.getcwd()
fullpath = os.path.join(cwd, 'image_classification/dataset/dataset_split')
dirname = os.path.join(cwd, 'image_classification/dataset/dataset_split')

def dir_to_list(dirname, path=os.path.pathsep):
    data = []
    for name in os.listdir(dirname):
        dct = {}
        dct['name'] = name
        dct['path'] = path + name

        full_path = os.path.join(dirname, name)
        if os.path.isfile(fullpath):
            dct['type'] = 'file'
        elif os.path.isdir(fullpath):
            dct['type'] = 'folder'
            dct['children'] = dir_to_list(full_path, path=path + name + os.path.pathsep)
    return data


csv_fname = 'data_splitting.json'
results_dir = './'
with open(os.path.join(results_dir, csv_fname), 'w') as f:
	f.write(dir_to_list(dirname, path=os.path.pathsep))

