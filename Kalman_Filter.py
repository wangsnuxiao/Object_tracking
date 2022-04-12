
import numpy as np 



class K_Filter:
    
    def __init__(self,center,v) :
        
        self.X=np.append(center,v) #state vector
        self.P=np.eye(4) #posteriori estimate covariance matrix
        self.firsttime=0

        

        pass

    def predict(self,dt):

        F=np.array([         #state-transition matrix
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]])

      
  

        Q=.01*np.eye(4)
        self.X=F.dot(self.X)
        self.P=F.dot(self.P).dot(F.T)+Q

        

        pass

    def predictoutput(self,dt): # return the prediction state 
            H=np.array([
                            [1,0,0,0],          #observation matrix
                            [0,1,0,0]
                        ])

            F=np.array([
                [1,0,dt,0],
                [0,1,0,dt],
                [0,0,1,0],
                [0,0,0,1]])

            
            return H.dot(F.dot(self.X))
            
            
    def set_state(self,centers,vs): # set the state and p matrix only once 
        if self.firsttime==0:
            self.X=np.append(centers,vs)
            self.P=np.eye(4)
            self.firsttime=1



       

    def update(self,z,R):  # update p and x using new measurment z
        H=np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])
        y=z-H.dot(self.X)  #z refer to measurment value 
        S=H.dot(self.P).dot(H.T)+R
        K=self.P.dot(H.T).dot(np.linalg.pinv(S))

        self.X=self.X+K.dot(y)
        self.P=(np.eye(4)-K.dot(H)).dot(self.P)
        







        pass



