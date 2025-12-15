import os

#重命名文件  双层目录
path='/data/gsd/3DSD/gce/a1.5b0/'

#获取该目录下所有文件，存入列表中
#
#dirs = os.listdir(path)
#for dir in sorted(dirs):
#    if dir[0] != '.':
#        dir_path = path + '/' +dir
#        names = os.listdir(dir_path)
#        for name in sorted(names):
#            if name[0] != '.' and len(name.split("_")[0]) == 1:
#                oldname = dir_path+ '/' + name
#                newname = dir_path + '/' + '0'+ name
#                os.rename(oldname,newname)
#            elif name.split("_")[2] == '0.png':
#                oldname = dir_path + '/' + name
#                os.remove( oldname )
#print('0')
#

dirs = os.listdir(path)
for dir in sorted(dirs):
    if dir[0] != '.':
        dir_path = path + '/' +dir
        names = os.listdir(dir_path)
        for name in sorted(names):
            if name.split("_")[0] == 'fake' and len(name.split("_")[1]) == 1:
                oldname = dir_path+ '/' + name
                newname = dir_path + '/' + name.split("_")[0] + '_0'+ name.split("ke_")[1]
                os.rename(oldname,newname)
            elif name.split("_")[0] == 'fake' and name.split("_")[3] == '0.png':
                oldname = dir_path + '/' + name
                os.remove( oldname )
print('0')




