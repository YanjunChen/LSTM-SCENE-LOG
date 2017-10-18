#need the sequence number
import sys
caffe_root = './caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
SEQ=128

#reshape label and produce clip_markers
class ReshapeLabel(caffe.Layer):   
    def setup(self, bottom, top):
        self.batch = bottom[0].data.shape[0]  #num videos processed per batch, predefined
        self.seq = SEQ   #length of processed clip
        self.top_names = ['reshape_label','clip_markers']
        #print self.top_names
            
        for top_index, name in enumerate(self.top_names):
            if name == 'reshape_label':
                shape = (self.seq, self.batch)
            elif name == 'clip_markers':
                shape = (self.seq, self.batch)
            top[top_index].reshape(*shape)                
                 
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        label_tmp = bottom[0].data.repeat(self.seq)
        label_r=label_tmp.reshape(self.batch,self.seq).T
        top[0].data[...] = label_r
        cm = np.ones_like(label_r)
        cm[0::self.seq] = 0
        top[1].data[...] = cm      
      
    def backward(self, top, propagate_down, bottom):
        pass

#transpose data from BATCH*SEQ*FEA to SEQ*BATCH*FEA
class DataTransposeSortMaxClip(caffe.Layer):
    
    def setup(self, bottom, top):
        self.batch = bottom[0].data.shape[0]  #num videos processed per batch
        self.fea= bottom[0].data.shape[1]
        self.len = bottom[0].data.shape[2]   #length of processed clip
	self.seq = SEQ
        shape = (self.seq, self.batch, self.fea)
        print 'setup',shape
        top[0].reshape(*shape)                
               
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        Feat = bottom[0].data.transpose([2,0,1])
        Feat_norm=np.sum(Feat,axis=2)
        Feat_sort=np.argsort(-Feat_norm,axis=0)
        Feat_new=np.zeros([self.len,self.batch,self.fea])
        for i in range(self.batch):
            Feat_new[:,i,:]=Feat[Feat_sort[:,i],i,:]
               
        top[0].data[...] = Feat_new[0:SEQ,:,:]
      
    def backward(self, top, propagate_down, bottom):
        Diff = np.zeros([self.len,self.batch,self.fea])
	Diff[0:SEQ,:,:] = top[0].diff
        Feat = bottom[0].data.transpose([2,0,1])
        Feat_norm=np.sum(Feat,axis=2)
        Feat_sort=np.argsort(-Feat_norm,axis=0)
        Feat_sort_index=np.argsort(Feat_sort,axis=0)
        Diff_rec=np.zeros([self.len,self.batch,self.fea])
        for i in range(self.batch):
	    Diff_rec[:,i,:]=Diff[Feat_sort_index[:,i],i,:]

        bottom[0].diff[...] = Diff_rec.transpose([1,2,0])

#transpose data from BATCH*SEQ*FEA to SEQ*BATCH*FEA
class DataTransposeSortRandomClip(caffe.Layer):
    
    def setup(self, bottom, top):
        self.batch = bottom[0].data.shape[0]  #num videos processed per batch
        self.fea= bottom[0].data.shape[1]
        self.len = bottom[0].data.shape[2]   #length of processed clip
	self.seq = SEQ
	PER_tmp=np.zeros([self.len,self.batch])
	for i in range(self.batch):
    	    per=np.random.permutation(self.len)
    	    PER_tmp[:,i]=per
	self.PER = PER_tmp.astype(int)
        shape = (self.seq, self.batch, self.fea)
        print 'setup',shape
        top[0].reshape(*shape)                
               
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        Feat = bottom[0].data.transpose([2,0,1])
        Feat_sort=self.PER
        Feat_new=np.zeros([self.len,self.batch,self.fea])
        for i in range(self.batch):
            Feat_new[:,i,:]=Feat[Feat_sort[:,i],i,:]
               
        top[0].data[...] = Feat_new[0:SEQ,:,:]
      
    def backward(self, top, propagate_down, bottom):
        Diff = np.zeros([self.len,self.batch,self.fea])
	Diff[0:SEQ,:,:] = top[0].diff
        Feat = bottom[0].data.transpose([2,0,1])
        Feat_sort=self.PER
        Feat_sort_index=np.argsort(Feat_sort,axis=0)
        Diff_rec=np.zeros([self.len,self.batch,self.fea])
        for i in range(self.batch):
	    Diff_rec[:,i,:]=Diff[Feat_sort_index[:,i],i,:]

        bottom[0].diff[...] = Diff_rec.transpose([1,2,0])

