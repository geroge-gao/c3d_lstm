import io

f=open('valid_rgb_list.txt','r')
fp = open('valid.txt', 'w+')

for line in f.readlines():
    path=line.split(' ')[0]
    framecnt=line.split(' ')[1]
    video_label=line.split(' ')[2]
    path=path.rstrip('/')
    path+='.avi'
    s=path+' '
    s+=framecnt+' '
    s+=video_label
    fp.writelines(s)

fp.close()
f.close()