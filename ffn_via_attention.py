import numpy as np

def matrix_unit(size, pos1, pos2):
    # returns the size-by-size array with a 1 in the (pos1, pos2) position
    return np.array([[int(i==pos1 and j==pos2) for i in range(size)] for j in range(size)])

def SiLU_W_QK_matrix(channel_to_focus, n_ctx, d_model, Omega=1000):
    # creates the W_QK matrix used for implementing SiLU with attention heads
    W_QK_1=2*Omega*sum([matrix_unit(d_model+n_ctx+1, n_ctx+d_model, i+d_model) for i in range(n_ctx+1)])
    W_QK_2=2*Omega*sum([matrix_unit(d_model+n_ctx+1, i+d_model, i+d_model) for i in range(n_ctx)])
    W_QK_3=-1*sum([matrix_unit(d_model+n_ctx+1, d_model+i, channel_to_focus) for i in range(n_ctx)])
    W_QK=W_QK_1+W_QK_2+W_QK_3
    return W_QK

def SiLU_W_OV_matrix(n_ctx, d_model, channel_to_focus):
    # creates the W_OV matrix used for implementing SiLU with attention heads
    return -1*matrix_unit(d_model+n_ctx+1, channel_to_focus, channel_to_focus)

def augment_classical_residual_stream(classical_residual_stream):
    #takes an n_ctx-by-d_model matrix, adds a row of 0s to the bottom, and a square identity matrix to the right
    #resulting matrix is (n_ctx+1)-by-(d_model+n_ctx+1) matrix
    n_ctx=classical_residual_stream.shape[0]
    d_model=classical_residual_stream.shape[1]
    zero_padding= np.zeros((1, d_model))
    identity_padding=np.identity(n_ctx+1)
    augmented_residual_stream=np.concatenate((np.concatenate((classical_residual_stream, zero_padding), axis=0)
        , identity_padding), axis=1)
    return augmented_residual_stream

def deaugment_residual_stream(augmented_residual_stream):
    # undoes the augmentation of the augment_classical_residual_stream method
    # takes an i-by-j matrix for j>i, removes the rightmost i columns and the bottom row
    number_of_columns_to_remove=augmented_residual_stream.shape[0]
    return augmented_residual_stream[0:-1, 0:-1*number_of_columns_to_remove]

def rowwise_softmax(arr):
    # applies softmax to each row of the matrix arr
    if len(arr.shape) > 1:
        for i in range(arr.shape[0]):
            arr[i]=rowwise_softmax(arr[i])
        return arr
    elif len(arr.shape) == 1:
        arr=arr-max(arr)
        arr=np.exp(arr)
        arr=arr/sum(arr)
        return arr

def SiLU_head_on_one_channel(classical_residual_stream, channel_to_focus, Omega=1000):
    # creates and applies an attention head which will apply SiLU to one dimension of the residual stream
    n_ctx=classical_residual_stream.shape[0]
    d_model=classical_residual_stream.shape[1]
    augmented_residual_stream=augment_classical_residual_stream(classical_residual_stream)
    W_QK=SiLU_W_QK_matrix(channel_to_focus, n_ctx, d_model, Omega=Omega)
    W_OV=SiLU_W_OV_matrix(n_ctx,d_model, channel_to_focus)
    head_output=attention_head(augmented_residual_stream, W_QK, W_OV)
    return head_output

def SiLU_to_all_channels(classical_residual_stream, Omega=1000):
    # creates and applies attention heads which apply SiLU to all dimensions of the residual stream
    augmented_residual_stream=augment_classical_residual_stream(classical_residual_stream)
    d_model=classical_residual_stream.shape[1]
    total_head_output=sum([SiLU_head_on_one_channel(classical_residual_stream, i, Omega=Omega) for i in range(d_model)])
    to_return=augmented_residual_stream+total_head_output
    return to_return

def direct_SiLU(x):
    # the SiLU function
    return x/(1+np.exp(-1*x))

def linear_transformation_with_augmentation(classical_residual_stream, weight_matrix, Omega=1000):
    # augments a residual stream to implement the weight matrix 
    n_ctx=classical_residual_stream.shape[0]
    d_model=classical_residual_stream.shape[1]
    augmented_residual_stream=augment_classical_residual_stream(classical_residual_stream)
    W_QK_augmented=matrix_corner_join(np.zeros((d_model,d_model)), 2*Omega*np.eye(n_ctx+1))
    W_OV_augmented=matrix_corner_join(weight_matrix-np.eye(d_model, d_model), np.zeros((n_ctx+1,n_ctx+1)))
    head_output=attention_head(augmented_residual_stream, W_QK_augmented, W_OV_augmented)
    return augmented_residual_stream+head_output

def normal_attention_with_augmentation(classical_residual_stream, W_QK_classical, W_OV_classical, Omega=1000):
    # augments a residual stream and the W_QK and W_OV matrices to produce the same result as if there was no augmentation
    n_ctx=classical_residual_stream.shape[0]
    augmented_residual_stream=augment_classical_residual_stream(classical_residual_stream)
    context_vector_attention= -2*Omega*sum([matrix_unit(n_ctx+1, n_ctx, pos2) for pos2 in range(n_ctx)])+2*Omega*matrix_unit(n_ctx+1, n_ctx, n_ctx)
    W_QK_augmented=matrix_corner_join(W_QK_classical, context_vector_attention)
    W_OV_augmented=matrix_corner_join(W_OV_classical, np.zeros((n_ctx+1,n_ctx+1)))
    head_output=attention_head(augmented_residual_stream, W_QK_augmented, W_OV_augmented)
    return augmented_residual_stream+head_output

def attention_head(residual_stream, W_QK, W_OV):
    # computes the output of an attention head
    # returns softmax(XQX^T)(XV), where X=residual_stream, Q=W_QK, and V=W_OV
    pre_attention=np.matmul(np.matmul(residual_stream, W_QK), np.transpose(residual_stream))
    attention=rowwise_softmax(pre_attention)
    return np.matmul(np.matmul(attention, residual_stream), W_OV)

def matrix_corner_join(A, B):
    #takes matrices A and B and returns the block matrix [A 0, 0 B]
    C=np.zeros((A.shape[0], B.shape[1]))
    D=np.zeros((B.shape[0], A.shape[1]))
    return np.block([[A, C], [D,B]])

def test_SiLU_function(test_residual_stream, Omega=1000, error_tolerance=10**-10):
    # compares result of entry-wise SiLU implemented:
    #   - directly
    #   - with attention heads
    # parameters:
    #   - test_residual_stream - an n_ctx-by-d_model array
    #   - Omega - the large number that overwhelms softmaxes
    #   - error_tolerance - the largest acceptable error in a single entry of the computed matrix
    attention_based_result=SiLU_to_all_channels(test_residual_stream, Omega=Omega)
    direct_result = np.vectorize(direct_SiLU)(test_residual_stream)
    augmented_direct_result=augment_classical_residual_stream(direct_result)
    largest_difference=np.amax(abs(attention_based_result-augmented_direct_result))
    if largest_difference<error_tolerance:
        print(f"The SiLU methods had nearly identical outputs! (Max error: {largest_difference})")
    else:
        print("BAD! VERY LARGE DIFFERENCE IN SILU OUTPUTS!")

def test_augmented_attention(test_residual_stream, test_W_QK, test_W_OV, Omega=1000, error_tolerance=10**-10):
    # compares result of an attention head implemented:
    #   - directly (as in a normal transformer)
    #   - directly (with the additional vector)
    # parameters:
    #   - test_residual_stream - an n_ctx-by-d_model array
    #   - test_W_QK - the d_model-by-d_model array encoding the attention pattern
    #   - test_W_OV - the d_model-by-d_model array encoding the output of a vector
    #   - Omega - the large number that overwhelms softmaxes
    #   - error_tolerance - the largest acceptable error in a single entry of the computed matrix
    attention_based_result=normal_attention_with_augmentation(test_residual_stream, test_W_QK, test_W_OV, Omega=Omega)
    direct_result = test_residual_stream+attention_head(test_residual_stream, test_W_QK, test_W_OV)
    augmented_direct_result=augment_classical_residual_stream(direct_result)
    largest_difference=np.amax(abs(attention_based_result-augmented_direct_result))
    if largest_difference<error_tolerance:
        print(f"The attention methods had nearly identical outputs! (Max error: {largest_difference})")
    else:
        print("BAD! VERY LARGE DIFFERENCE IN ATTENTION OUTPUTS!")

def test_linear(test_residual_stream, test_W, Omega=1000, error_tolerance=10**-10):
    # compares result of a linear transformation implemented:
    #   - directly (matrix multiplication)
    #   - with attention heads
    # parameters:
    #   - test_residual_stream - an n_ctx-by-d_model array
    #   - test_W - a d_model-by-d_model array encoding the linear transformation
    #   - Omega - the large number that overwhelms softmaxes
    #   - error_tolerance - the largest acceptable error in a single entry of the computed matrix
    attention_based_result=linear_transformation_with_augmentation(test_residual_stream, test_W, Omega=Omega)
    direct_result = np.matmul(test_residual_stream, test_W)
    augmented_direct_result=augment_classical_residual_stream(direct_result)
    largest_difference=np.amax(abs(attention_based_result-augmented_direct_result))
    if largest_difference<error_tolerance:
        print(f"The linear methods had nearly identical outputs! (Max error: {largest_difference})")
    else:
        print("BAD! VERY LARGE DIFFERENCE IN LINEAR OUTPUTS!")


if __name__=="__main__":
    n_ctx=20
    d_model=30
    Omega=1000
    error_tolerance=10**-10
    for _ in range(100):
        test_residual_stream=np.random.normal(size=(n_ctx,d_model))
        test_W_linear=np.random.normal(size=(d_model,d_model))
        test_linear(test_residual_stream, test_W_linear, Omega=Omega, error_tolerance=error_tolerance)
    for _ in range(100):
        test_residual_stream=np.random.normal(size=(n_ctx,d_model))
        test_W_QK=np.random.normal(size=(d_model,d_model))
        test_W_OV=np.random.normal(size=(d_model,d_model))
        test_augmented_attention(test_residual_stream, test_W_QK, test_W_OV, Omega=Omega, error_tolerance=error_tolerance)
    for _ in range(100):
        test_residual_stream=np.random.normal(size=(n_ctx,d_model))
        test_SiLU_function(test_residual_stream, Omega=Omega, error_tolerance=error_tolerance)
