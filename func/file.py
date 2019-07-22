from __future__ import print_function
import os, sys
import shutil


def removedir(folder):
	filelist=[]
	rootdir=folder                     
	filelist=os.listdir(rootdir)                
	for f in filelist:
    		filepath = os.path.join(rootdir,f)   
    		if os.path.isfile(filepath):            
        		os.remove(filepath)                
        		print(str(filepath)+' removed!')
    		elif os.path.isdir(filepath):
        		shutil.rmtree(filepath,True)       
        		print('dir '+str(filepath)+' removed!')
	shutil.rmtree(rootdir,True)             
	print('dir '+folder+' removed!')
