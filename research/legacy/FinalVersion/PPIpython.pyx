from __future__ import division
from PIL import Image
#import Image
import numpy as np
cimport numpy as np
import math
import time
import matplotlib.pyplot as plt
import cProfile
import pstats

#from libc.math cimport abs

cimport cython

DTYPE = np.int
ctypedef np.int_t DTYPE_t

class cone_H(object):
    def __init__(self, angle_min, angle_max):
        self.angle_min = angle_min 
        self.angle_max = angle_max 

class cone_B(object):
    def __init__(self, angle_min, angle_max):
        self.angle_min = angle_min
        self.angle_max = angle_max

class cone_E(object):
    def __init__(self, angle_min, angle_max):
        self.angle_min = angle_min
        self.angle_max = angle_max

class cone_W(object):
    def __init__(self, angle_min, angle_max):
        self.angle_min = angle_min
        self.angle_max = angle_max

def display_image(image):
    plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()

def frange(start, end, step):
    while (start > end):
        yield start
        start += step
                    


@cython.boundscheck(False)
def bresenham_line(int x, int y, int x2, int y2):
    """Brensenham line algorithm"""
    """ To get the coordinates of a line segment to create the filter prototypes """
    cdef int steep, dx, dy, sx, sy, d, i
    steep = 0
    coords = []
    dx = abs(x2 - x)
    if (x2 - x) > 0: sx = 1
    else: sx = -1
    dy = abs(y2 - y)
    if (y2 - y) > 0: sy = 1
    else: sy = -1
    if dy > dx:
        steep = 1
        x,y = y,x
        dx,dy = dy,dx
        sx,sy = sy,sx
    d = (2 * dy) - dx
    for i in range(0,dx):
        if steep: coords.append((abs(y),abs(x)))
        else: coords.append((abs(x),abs(y)))
        while d >= 0:
            y = y + sy
            d = d - (2 * dx)
        x = x + sx
        d = d + (2 * dy)
    coords.append((abs(x2),abs(y2)))
    return coords

@cython.boundscheck(False)
def create_cone_endpoints(float angle_min1, float angle_max1, int length):
    cdef int leng, p1, p2, angle_min, angle_max, angle_cen, ind, indic, angle_ind, pointx, pointy, endpoint_x, endpoint_y
    cdef float angle, minx, miny, maxy, maxx,
    end_point = []
    angle_min=int(round((180*angle_min1)/np.pi))
    angle_max=int(round((180*angle_max1)/np.pi))
    angle_cen=int((angle_max-angle_min)/2)
    angle_cen=angle_min+angle_cen
    angle=(angle_cen*np.pi)/180
    pointx=int(round((length*(-math.sin(angle)))))
    pointy=int(round((length*math.cos(angle))))
    if(pointx!=0): ind=abs(pointx)
    else: ind=abs(pointy)
    for indic in range(-ind, 0):
        if((pointx<0)|(pointy>0)):angle_ind=-indic
        else: angle_ind=indic
        if(pointx==0):endpoint_x=pointx+angle_ind 
        else: endpoint_x=pointx
        if(pointy==0):endpoint_y=pointy+angle_ind
        else: endpoint_y=pointy
        end_point.append(((endpoint_x),(endpoint_y)))
    for indic in range(0, abs(ind)+1):
        if((pointx<0)or(pointy>0)):angle_ind=-indic
        else: angle_ind=indic
        if(pointx==0):endpoint_x=pointx+angle_ind 
        else: endpoint_x=pointx
        if(pointy==0):endpoint_y=pointy+angle_ind
        else: endpoint_y=pointy
        end_point.append(((endpoint_x),(endpoint_y)))

    return end_point

@cython.boundscheck(False)
def compute_image_sum(np.ndarray[np.uint8_t, ndim=2] potential_image1 not None, int start_pointX, int start_pointY, int end_pointX, int end_pointY):
    cdef int lengt, pixel1, pixel2
    cdef float output_sum
    coordinates =  bresenham_line(start_pointX,start_pointY,end_pointX,end_pointY)[1:]
    output_sum = 0.0
    lengt = len(coordinates)
    for pixel_ind in range(0, lengt):
        pixel1, pixel2 = coordinates[pixel_ind]
        output_sum += potential_image1[pixel1, pixel2]
    
    return output_sum



@cython.boundscheck(False)
def draw_path_random(np.ndarray[np.float_t, ndim=2] image_cost, np.ndarray[DTYPE_t, ndim=4] array_coordinates):
    cdef int rows, cols, ii, jj, iii
    rows, cols = np.shape(image_cost)
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    image[:,:,0]=255
    image[:,:,1]=255
    image[:,:,2]=255
    
    for iii in range(0,290):
        #print "columnas", cols
        ii=np.random.randint(0,160)
        jj=np.random.randint(0,cols-1)
        #print "aleatorio", ii
        draw_RGB(image, (ii,jj), image_cost, array_coordinates)
        
    for ii1 in range(0,20):
        i1=np.random.randint(58,114)
        j1=np.random.randint(42,88)
        draw_RGB(image, (i1,j1), image_cost, array_coordinates)

    return image   


def draw_RGB(image, ind_tup, image_cost, array_coordinates):
    i, j = ind_tup
    c0=np.random.randint(0,255)
    c1=np.random.randint(0,255)
    c2=np.random.randint(0,255)
    color=[c0,c1,c2]
    
    nb_segments = array_coordinates.shape[2]
    precedent_point  = (i,j)
    if(image_cost[i,j]!=float('inf')):
        for index in range(0, nb_segments):
            current_point = array_coordinates[i,j, index, [0,1]]
            draw_segmentRGB(precedent_point, current_point, image, index, color)
            precedent_point = current_point 


def draw_segmentRGB(precedent_point, current_point, image, ind, color):
    coordinates1 =  bresenham_line(precedent_point[0],precedent_point[1],current_point[0],current_point[1])
    if (ind!=0):
        del coordinates1[0]
    for pixel in coordinates1:
        image[pixel[0],pixel[1],0]=color[0]
        image[pixel[0],pixel[1],1]=color[1]
        image[pixel[0],pixel[1],2]=color[2]
  
    
    
def draw_segment(precedent_point, current_point, image, color):
    """Test function: draw a segment"""
    coordinates1 =  bresenham_line(precedent_point[0],precedent_point[1],current_point[0],current_point[1])
    #print "k pasa", coordinates1
    for pixel in coordinates1:
        image[pixel]=color


def draw_path(image, ind_tup,image_cost, array_coordinates, color):
    i, j = ind_tup
    nb_segments = array_coordinates.shape[2]
    precedent_point  = (i,j)
    if(image_cost[i,j]!=float('inf')):
        for index in range(0, nb_segments):
            current_point = array_coordinates[i,j, index, [0,1]]
            draw_segment(precedent_point, current_point, image, color)
            precedent_point = current_point

    return image

    
@cython.boundscheck(False)
def voting_without_tortuosity(np.ndarray[np.float_t, ndim=2] image_cost_global1 not None, np.ndarray[DTYPE_t, ndim=4] array_coord not None, float tortuosityMin):   
    
    cdef int rows, cols, nb_segment, i, j, index, V1_X, V1_Y, V2_X, V2_Y
    cdef float tortuosity, Vfinal, norm1, norm2
    cdef np.ndarray[np.float_t, ndim=2] image_cost_global
    rows, cols = np.shape(image_cost_global1)
    image_cost_global = np.zeros((rows, cols), dtype=np.float)
    nb_segment = array_coord.shape[2]
    image_cost_global[::]=image_cost_global1[::]
    for i in range(0, rows):
        for j in range(0, cols):
            tortuosity=1.0
            if(image_cost_global[i,j]!=float('inf')):
                for index in range(1, nb_segment-1):
                    V1_X=array_coord[i,j, index, 1] - array_coord[i,j, index-1, 1]
                    V1_Y=array_coord[i,j, index, 0] - array_coord[i,j, index-1, 0]
                    V2_X=array_coord[i,j, index+1, 1] - array_coord[i,j, index, 1]
                    V2_Y=array_coord[i,j, index+1, 0] - array_coord[i,j, index, 0]
                    Vfinal=(V1_X*V2_X)+(V1_Y*V2_Y)
                    norm1=math.sqrt(V1_X**2+V1_Y**2)
                    norm2=math.sqrt(V2_X**2+V2_Y**2)
                    tortuosity*=Vfinal/(norm1*norm2)
                   # print"vector1: ", V1_X,V1_Y
                   # print"vector2: ",V2_X,V2_Y
                    #print"norm1 y norm2: ", norm1, norm2
                    #print "putno", i,j
                if (tortuosity<tortuosityMin):
                    image_cost_global[i,j]=float('inf')

       
    return image_cost_global


@cython.boundscheck(False)
def voting(np.ndarray[np.float_t, ndim=2] image_cost_global not None, np.ndarray[DTYPE_t, ndim=4] array_coordinates not None):
    
    cdef int rows, cols, i, j, index, pixel, pixel1, pixel2, max_value, leng, nb_segment, precedent_pointX, precedent_pointY, current_pointX,current_pointY
    cdef np.ndarray[DTYPE_t, ndim=2] image
    cdef np.ndarray[DTYPE_t, ndim=2] imageInverted
    rows, cols = np.shape(image_cost_global)
    image = np.zeros((rows, cols), dtype=np.int)
    imageInverted = np.zeros((rows, cols), dtype=np.int)
    nb_segment = array_coordinates.shape[2]
    max_value=0
    
    for i in range(0, rows):
        for j in range(0, cols):
            precedent_pointX, precedent_pointY   = (i,j)
            if(image_cost_global[i,j]!=float('inf')):
                for index in range(0, nb_segment):
                    current_pointX = array_coordinates[i,j, index, 0]
                    current_pointY = array_coordinates[i,j, index, 1]
                    if(index==0):
                        coordinates= bresenham_line(precedent_pointX, precedent_pointY,current_pointX,current_pointY)
                    else:
                        coordinates= bresenham_line(precedent_pointX, precedent_pointY,current_pointX,current_pointY)[1:]
                    leng=len(coordinates)
                    for pixel in range(0, leng):
                        pixel1, pixel2=coordinates[pixel]
                        image[pixel1, pixel2]+=1
                        if image[pixel1, pixel2]>max_value:
                            max_value=image[pixel1, pixel2]
                    precedent_pointX, precedent_pointY = current_pointX, current_pointY
    
    image[0:2,0:cols]=0
    image[rows-2:rows,0:cols]=0
    image[0:rows,0:2]=0
    image[0:rows,cols-2:cols]=0   
    
    imageInverted[::]=max_value-image[::]
                        
    return (image, imageInverted)


@cython.boundscheck(False)
def orientation(np.ndarray[np.uint8_t, ndim=2] image not None, np.ndarray[np.float_t, ndim=2] image_cost_global not None, np.ndarray[DTYPE_t, ndim=4] array_coord not None):

    cdef int rows, cols, ii, jj, i, j, x, y,index, p, pixel, pixel1, pixel2, direcc1, direcc2, found, ant, leng, nb_segment, precedent_pointX, precedent_pointY, current_pointX,current_pointY, t, t1, t2
    cdef float T1, T2, angle, alfa
    rows, cols = np.shape(image_cost_global)
    nb_segment = array_coord.shape[2]
    
    for ii in xrange(0, rows, 4):
        for jj in xrange(0, cols, 4):
            T1=0
            T2=0
            for i in xrange(ii-30,ii+30,2):
                for j in xrange(jj-30,jj+30,2):
                    if(((i>=0)&(i<=(rows-1))) & ((j>=0)&(j<=(cols-1)))):
                        precedent_pointX, precedent_pointY   = (i,j)
                        found=0
                        if(image_cost_global[i,j]!=float('inf')):
                            for index in range(0, nb_segment):
                                current_pointX = array_coord[i,j, index, 0]
                                current_pointY = array_coord[i,j, index, 1]
                                if(index==0):
                                    coordinates= bresenham_line(precedent_pointX, precedent_pointY,current_pointX,current_pointY)
                                else:
                                    coordinates= bresenham_line(precedent_pointX, precedent_pointY,current_pointX,current_pointY)[1:]
                                leng=len(coordinates)
                                for pixel in range(0, leng):
                                    pixel1, pixel2=coordinates[pixel]
                                    if (((pixel1 == ii) & (pixel2==jj))):
                                        p=3
                                        if(index==0):
                                            t1=array_coord[i,j, index, 0]-i
                                            t2=array_coord[i,j, index, 1]-j
                                        else:
                                            t1=array_coord[i,j, index, 0]-array_coord[i,j, index-1, 0]
                                            t2=array_coord[i,j, index, 1]-array_coord[i,j, index-1, 1]
                                        angle=((t1*3)+(t2*0))/(math.sqrt(t1**2+t2**2)*3)       
                                        angle=math.acos(angle)
                                        t=(t1*t1)+(t2*t2)
                                        T1+=t*math.sin(2*angle)
                                        T2+=t*math.cos(2*angle)
                                        found=1
                                        break
                                if (found==1): break        
                                precedent_pointX, precedent_pointY = current_pointX, current_pointY
    
            alfa=math.atan(T1/T2)/2
            x=0
            y=0
            direcc1=ii-array_coord[ii,jj, nb_segment-1, 0]
            if(direcc1<0): alfa=-alfa #the path goes down,in the opposite direction taken as reference
            if(alfa<0): alfa=np.pi+alfa 
            alfa=(alfa*180)/np.pi
            alfa=math.ceil(alfa)

            if ((abs(alfa)>=0) & (abs(alfa)<11)) :x=2
    
            if ((abs(alfa)>=11) & (abs(alfa)<45)):
                x=2
                y=-1
            if ((abs(alfa)>=45) & (abs(alfa)<46)):
                x=2
                y=-2
            if ((abs(alfa)>=46) & (abs(alfa)<90)):
                x=1
                y=-2
            if ((abs(alfa)>=90) & (abs(alfa)<91)): y=-2
            if ((abs(alfa)>=91) & (abs(alfa)<135)):
                x=-1
                y=2
            if ((abs(alfa)>=135) & (abs(alfa)<157)):
                x=-2
                y=-2
            if ((abs(alfa)>=136) & (abs(alfa)<170)):
                x=-2
                y=-1
            if ((abs(alfa)>=170) & (abs(alfa)<=180)): x=-2
    
                   
            if((direcc1<0)&(y!=0)):
                if(x!=0): plt.arrow(jj,ii,-x,-y,fc="b", ec="b",lod=True,head_width=1, head_length=0.5)
                else: 
                    if(x==0): plt.arrow(jj,ii,x,-y,fc="b", ec="b",lod=True,head_width=1, head_length=0.5)
            else: plt.arrow(jj,ii,x,y,fc="b", ec="b",lod=True,head_width=1, head_length=0.5)

    plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    
    return image



@cython.boundscheck(False)
def pruning(np.ndarray[np.float_t, ndim=2] image_cost_global not None, np.ndarray[DTYPE_t, ndim=4] array_coordinates not None, int l, float porcent, int d):

    cdef int rows, cols, nb_segment, m, pointminX, pointminY, i, j, current_vertexX, current_vertexY, pathmin_vertex_X, pathmin_vertex_Y, num_vertex
    cdef float dist, pathmin, min, porcent_vertex
    cdef np.ndarray[np.float_t, ndim=2] imageStop
    cdef np.ndarray[np.float_t, ndim=2] image_cost_global1
    cdef np.ndarray[np.float_t, ndim=2] imprueba
    cdef np.ndarray[np.uint8_t, ndim=3] image
    rows, cols = np.shape(image_cost_global)
    imageStop = np.zeros((rows, cols), dtype=np.float)
    image_cost_global1 = np.zeros((rows, cols), dtype=np.float)
    imprueba = np.ones((rows, cols), dtype=np.float)
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    nb_segment = array_coordinates.shape[2]
    tam=(l*nb_segment)-(porcent*nb_segment)*l
    image_cost_global1[::]=image_cost_global[::]
    image[:,:,0]=255
    image[:,:,1]=255
    image[:,:,2]=255
    prueb1=[]


    prueb1=image_cost_global==float('inf')
    imageStop=np.where(prueb1,imprueba,imageStop)
    while(imageStop.all()!=True):
        min=float('inf')
        for m in range(0, rows):
            pathmin=np.amin(image_cost_global1[m,:])
            point=np.where(image_cost_global1==pathmin)
            if (pathmin<min):
                min=pathmin
                pointminX=point[0][0]
                pointminY=point[1][0]
        imageStop[pointminX,pointminY]=1
        image_cost_global1[pointminX,pointminY]=float('inf')
        for i in range(pointminX-tam,pointminX+tam):
            for j in range(pointminY-tam,pointminY+tam):
                num_vertex=0
                porcent_vertex=0.0
                if(((i>=0)&(i<=(rows-1))) & ((j>=0)&(j<=(cols-1)))):
                    if(image_cost_global1[i,j]!=float('inf')):
                        for index in range(0, nb_segment):
                            current_vertex_X=array_coordinates[i,j, index, 0]
                            current_vertex_Y=array_coordinates[i,j, index, 1]
                            for index2 in range(0, nb_segment):
                                pathmin_vertex_X=array_coordinates[pointminX,pointminY, index2, 0]
                                pathmin_vertex_Y=array_coordinates[pointminX,pointminY, index2, 1]
                                dist=math.sqrt((current_vertex_X-pathmin_vertex_X)**2+(current_vertex_Y-pathmin_vertex_Y)**2)
                                if (dist<d):
                                    num_vertex+=1
                                    break
                        porcent_vertex=num_vertex/nb_segment
                        if (porcent_vertex>=porcent):
                            image_cost_global[i,j]=float('inf')
                            image_cost_global1[i,j]=float('inf')
                            imageStop[i,j]=1
                    
                  
    for ii in range(0,rows):
        for jj in range(0,cols):
            if(image_cost_global[ii,jj]!=float('inf')): 
                draw_RGB(image, (ii,jj), image_cost_global, array_coordinates)
    
    return image 



@cython.boundscheck(False)
def select_best_result_among_cone(shortest_path_and_image_list, int nb_segments):
    cdef int rows, cols, i, j, iter, imag
    rows, cols = np.shape(shortest_path_and_image_list[0][0])
    cdef np.ndarray[np.float_t, ndim=2] image_cost_global
    image_cost_global=np.zeros((rows, cols), dtype=np.float)
    array_coordinates_global=np.zeros((shortest_path_and_image_list[0][0].shape[0], shortest_path_and_image_list[0][0].shape[1], nb_segments, 2)).astype(int)
    for i in range(0, rows):
        for j in range(0, cols):
            minim_cost_global=float('inf')
            cost_inf=True
            for imag in range(0, len(shortest_path_and_image_list)):
                if (shortest_path_and_image_list[imag][0][i,j] < minim_cost_global):
                    minim_cost_global=shortest_path_and_image_list[imag][0][i,j]
                    image_cost_global[i,j]=minim_cost_global
                    cost_inf=False
                    for iter in range(0, nb_segments):
                        array_coordinates_global[i,j,iter,[0,1]]=shortest_path_and_image_list[imag][1][i,j,iter,[0,1]]
                if(cost_inf==True):
                    image_cost_global[i,j]=float('inf')


    return (image_cost_global, array_coordinates_global)


@cython.boundscheck(False)
def compute_shortest_path_in_cone(np.ndarray[np.uint8_t, ndim=2] potential_image not None, np.ndarray[DTYPE_t, ndim=4] array_coordinates not None, np.ndarray[np.float_t, ndim=3] array_sum not None, coneEndpoints, int nb_segment):
    cdef int rows, cols, leng, i, j, iteration, l, num_endpoints, start_pointX, start_pointY, end_pointX, end_pointY
    cdef float cost, minimal_cost
    rows, cols = np.shape(potential_image)
    cdef np.ndarray[np.float_t, ndim=2] image_cost1
    image_cost1 = np.zeros((rows, cols), dtype=np.float)
    # image_cost contains the cost of the segment that yielded the shortest path
    cdef np.ndarray[np.float_t, ndim=2] image_cost
    image_cost = np.zeros((rows, cols), dtype=np.float)
    # array_coord contains the coordinates of the end point that yielded the shortest path
    #define cone end points
    #coneEndpoints = create_cone_endpoints(cone.angle_min, cone.angle_max, length)
    #array_sum contains, for each pixel, the sum of the line between the start_point and its ends_points
    
    leng=len(coneEndpoints)
    for iteration in range(0, nb_segment):
        image_cost1[:,:]=image_cost[:,:]
        for i in range(0, rows):
            for j in range(0, cols):
                start_pointX, start_pointY = (i,j)
                minimal_cost = float('inf')
                num_endpoints=0
                cost_inf=True
                for deplacement_vector in range(0, leng):
                    deplacement_vector1, deplacement_vector2 = coneEndpoints[deplacement_vector] 
                    end_pointX = start_pointX + deplacement_vector1
                    end_pointY = start_pointY + deplacement_vector2
                    if(((end_pointX>=0)&(end_pointX<=(rows-1))) & ((end_pointY>=0)&(end_pointY<=(cols-1)))):
                        if(iteration==0):
                            sum= compute_image_sum(potential_image, start_pointX, start_pointY, end_pointX, end_pointY)
                            cost = sum + image_cost1[end_pointX, end_pointY]
                            array_sum[i,j,num_endpoints]=sum
                        else: cost = array_sum[i,j,num_endpoints] + image_cost1[end_pointX, end_pointY]
                        
                        if ((cost<float('inf'))&(cost <= minimal_cost)):
                            minimal_cost = cost
                            cost_inf=False
                            if(iteration==0):
                                array_coordinates[i,j, iteration, 0] = end_pointX
                                array_coordinates[i,j, iteration, 1] = end_pointY
                            if(iteration!=0):
                                array_coordinates[i,j,0,0]=end_pointX
                                array_coordinates[i,j,0,1]=end_pointY
                            
                                for l in range(1, iteration):
                                    array_coordinates[i,j,l,0]=array_coordinates[end_pointX,end_pointY,l-1,0]
                                    array_coordinates[i,j,l,1]=array_coordinates[end_pointX,end_pointY,l-1,1]
                                   
                                array_coordinates[i,j, iteration, 0] = array_coordinates[end_pointX,end_pointY, iteration-1, 0]
                                array_coordinates[i,j, iteration, 1] = array_coordinates[end_pointX,end_pointY, iteration-1, 1]
                                
                            image_cost[i,j]=cost
                                    
                    if((cost_inf==True)):
                        image_cost[i,j]=float('inf')
                        array_coordinates[i,j, iteration, 0] = -1
                        array_coordinates[i,j, iteration, 1] = -1
                    num_endpoints+=1

    
    
    return (image_cost, array_coordinates)




def compute_ppi(potential_image, segment_length, nb_segment):
    #define cone bounds
    
    cone_H.angle_min = np.pi/4
    cone_H.angle_max = cone_H.angle_min + np.pi/2
    cone_B.angle_min = np.pi + np.pi/4
    cone_B.angle_max = cone_B.angle_min + np.pi/2
    cone_E.angle_min = -np.pi/4
    cone_E.angle_max = cone_E.angle_min + np.pi/2
    cone_W.angle_min = 3*np.pi/4
    cone_W.angle_max = cone_W.angle_min + np.pi/2
    
    cone_list = [cone_H, cone_B, cone_E, cone_W] # add diagonal cones
    #cone_list = [cone_B]
    #compute shortest path in each cone
    shortest_path_and_image_list = [] # [(image_cost, array[(x,y)])]
  
    for cone in cone_list :
        coneEndpoints = create_cone_endpoints(cone.angle_min, cone.angle_max, segment_length)
        array_coordinates = np.zeros((potential_image.shape[0], potential_image.shape[1], nb_segment, 2)).astype(int)
        array_sum=np.zeros((potential_image.shape[0], potential_image.shape[1],len(coneEndpoints))).astype(float)
        shortest_path_and_image_list.append((compute_shortest_path_in_cone(potential_image, array_coordinates, array_sum, coneEndpoints, nb_segment)))
       
    #retain best path among all cones at each pixel
    shortest_path_and_image=[] # [(image_cost_global_best_resul, array_best_result[(x,y)])]
            
    shortest_path_and_image = select_best_result_among_cone(shortest_path_and_image_list, nb_segment)

    return shortest_path_and_image
    

def noisy2D(signal, sigma = 0.25):
    return signal + sigma*np.random.randn(signal.shape[0], signal.shape[1])

def example(ImageName,segment_length=3, nb_segment=10, test=True, tortuosityMin=0.75, Voting=1, without_tortousity=1, Orientation=1, Pruning=1, porcent=0.4, d=10, decimat=1, debug=0):
    """ Test example: return a numpy array of the voting """
    im = Image.open(ImageName)
    imarray = np.matrix(im)
    rows = imarray.shape[0]
    cols= imarray.shape[1]
    print "Filas,columnas", rows, cols
    #imarray=imarray[0:imarray.shape[0]:decimat,0:imarray.shape[1]:decimat]
    display_image(imarray)
    print "Compute the ppi"
    start = time.time()
    ppi = compute_ppi(imarray, segment_length, nb_segment)
    elapsed = (time.time() - start)
    print "Time for ppi:", elapsed
    
        
    if (test):
        display_image(ppi[0])
        np.save("imageCost",ppi[0])
        data = np.load("imageCost.npy")
        img = Image.fromarray(data)
        img.save("imageCost.tif")
    
        circ=draw_path_random(ppi[0], ppi[1])
        display_image(circ)
        np.save("imagePath",circ)
        data = np.load("imagePath.npy")
        img = Image.fromarray(data)
        img.save("imagePath.tif")
        
#draw_random_path(imarray, ppi)
# display_image(imarray)
    #display_image(ppi[0])
    #    point = (imarray.shape[0]/2, imarray.shape[1]/2+5)
    #    center = (imarray.shape[0]/2, imarray.shape[1]/2)
    #    display_image(imarray)
    #    for offset in [(-2, -2), (-2, 2), (-2, 0), (0, -2), (0, 2), (2, 0)]:
    #    draw_path(imarray,(75, 100), ppi[0], ppi[1], 0)
    #    draw_path(imarray,(75, 150), ppi[0], ppi[1], 0)
    #    draw_path(imarray,(100, 100), ppi[0], ppi[1], 0)
    #    display_image(imarray)
    print "Compute the path voting image"
    #imp=draw_path(imarray, (30,30),ppi[0], ppi[1], 255)
    #display_image(imp)
   # imp=draw_path(imarray, (60,60),ppi[0], ppi[1], 255)
   # display_image(imp)
    if(Voting):
        if(without_tortousity): 
            start = time.time()
            votet = voting_without_tortuosity(ppi[0], ppi[1], tortuosityMin)
            (vote, voteInver) = voting(votet, ppi[1])
            elapsed = (time.time() - start)
            print "Time for voting without tortuosity:", elapsed
            display_image(vote)
            display_image(voteInver)
            np.save("imageVoteInver",voteInver)
            np.save("imageVote",vote)
        if(without_tortousity==0):
            (vote, voteInver) = voting(ppi[0], ppi[1])
            elapsed = (time.time() - start)
            print "Time for voting:", elapsed
            display_image(vote)
            display_image(voteInver)
            np.save("imageVoteInver",voteInver)
            np.save("imageVote",vote)
    print "Compute the Orientation"
    if (Orientation):
        image=orientation(imarray,ppi[0], ppi[1])
        display_image(image)
    if(Pruning):
        print "Compute the pruning:"
        if(without_tortousity):
            imp = voting_without_tortuosity(ppi[0], ppi[1], tortuosityMin)
            imPruning=pruning(imp,ppi[1], segment_length, porcent, d)
        if(without_tortousity!=1):
            imPruning=pruning(ppi[0],ppi[1], segment_length, porcent, d)
        display_image(imPruning)

    #return voteInverted

def profile_example(ImageName="guiawires.pgm", segment_length=3, nb_segment=10, debug=0, statname = 'ppi_stats', showMe=0):
    cProfile.runctx('example(ImageName, segment_length, nb_segment, showMe, debug)',  globals(), locals(), statname)
    p = pstats.Stats(statname)
    p.strip_dirs().sort_stats('time', 'cum').print_stats(50)
    
    