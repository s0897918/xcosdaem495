import math

# def optimizer(l_target, idx=0):
    
#     w_grad=[[0.088, 0.104], [0.176, 0.208]]
#     x=[0.2, 0.4]
#     w=[[0.1, 0.5], [-0.3, 0.8]]
#     q1=w[0][0]*x[0]+w[0][1]*x[1]
#     q2=w[1][0]*x[0]+w[1][1]*x[1]
#     l=math.pow(q1,2) + math.pow(q2,2)
#     print ("l:", l)

#     grad = w_grad[0][0]
    
#     w_target = (l_target-l)/grad + w[0][0]
#     print ("w_target:", w_target)

def grad_compute(delta=0, para='w', idx=0):
    w=[[0.1, 0.5], [-0.3, 0.8]]
    w_grad=[[0.088, 0.104], [0.176, 0.208]]

    x=[0.2, 0.4]
    q1=w[0][0]*x[0]+w[0][1]*x[1]
    q2=w[1][0]*x[0]+w[1][1]*x[1]
    l=math.pow(q1,2) + math.pow(q2,2)
    print ("l:", l)

    if (para == 'w'):
        if (idx == 0):
            w[0][0] += delta
        elif (idx==1):
            w[0][1] += delta
        elif (idx == 2):
            w[1][0] += delta
        elif (idx == 3):
            w[1][1] += delta
        else:
            print("Error w idx")
            return
        
    elif (para == 'x'):
        if (idx == 0):
            x[0] += delta
        elif (idx == 1):
            x[1] += delta
        else:
            print("Error x idx")
            return
    else:
        print("Error para value")
        return
    

    q1_new = w[0][0]*x[0]+w[0][1]*x[1]
    q2_new = w[1][0]*x[0]+w[1][1]*x[1]
    l_new=math.pow(q1_new, 2) + math.pow(q2_new, 2)
    print ("l_new:", l_new)

    if (delta != 0):
        grad = (l_new - l)/delta
        

    print (para, "grad:{:.3f}".format(grad))
    print ("")

    
def get_loss(target_l, x, w):
    q1=w[0][0]*x[0]+w[0][1]*x[1]
    q2=w[1][0]*x[0]+w[1][1]*x[1]
    current_l=math.pow(q1,2) + math.pow(q2,2)
    return target_l-current_l, current_l

def update_weight(target_l, current_l, grad_w, w):
    grad = grad_w[0][0]
    w[0][0] += (0.1)* ((target_l-current_l)/grad)
    print ("new w:", w[0][0])

    grad = grad_w[0][1]
    w[0][1] += (0.1)* ((target_l-current_l)/grad)
    print ("new w:", w[0][1])

    grad = grad_w[1][0]
    w[1][0] += (0.1)* ((target_l-current_l)/grad)
    print ("new w:", w[1][0])

    grad = grad_w[1][1]
    w[1][1] += (0.1)* ((target_l-current_l)/grad)
    print ("new w:", w[1][1])
    
    return w

def optimizer(target_l, idx=0):

    print (" target l: ", target_l)
    grad_w=[[0.088, 0.104], [0.176, 0.208]]
    
    x=[0.2, 0.4]
    w=[[0.1, 0.5], [-0.3, 0.8]]
    loss, current_l = get_loss(target_l, x, w)

    w = update_weight(target_l, current_l, grad_w, w)

    loss, current_l = get_loss(target_l, x, w)

    #update w gradients
    grad_o = [2*q1, 2*q2]
    grad_w[0][0] = grad_o[0]*x[0]
    grad_w[0][1] = grad_o[0]*x[1]
    grad_w[1][0] = grad_o[1]*x[0]
    grad_w[1][1] = grad_o[1]*x[1]

    #update w
    grad = grad_w[0][0]
    w[0][0] += (0.1)* ((target_l-current_l)/grad)
    print ("new w:", w[0][0])

    grad = grad_w[0][1]
    w[0][1] += (0.1)* ((target_l-current_l)/grad)
    print ("new w:", w[0][1])

    grad = grad_w[1][0]
    w[1][0] += (0.1)* ((target_l-current_l)/grad)
    print ("new w:", w[1][0])

    grad = grad_w[1][1]
    w[1][1] += (0.1)* ((target_l-current_l)/grad)
    print ("new w:", w[1][1])

    q1=w[0][0]*x[0]+w[0][1]*x[1]
    q2=w[1][0]*x[0]+w[1][1]*x[1]
    current_l=math.pow(q1,2) + math.pow(q2,2)
    print ("current_l:", current_l)

    

if __name__ == '__main__':

    optimizer(1, 0)
    # delta = 0.0001
    # grad_compute(delta, 'w', 0)    
    # grad_compute(delta, 'w', 1)
    # grad_compute(delta, 'w', 2)
    # grad_compute(delta, 'w', 3)
    
    # grad_compute(delta, 'x', 0)
    # grad_compute(delta, 'x', 1)

