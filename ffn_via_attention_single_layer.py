import numpy as np

def matrix_unit(size_1, pos1, pos2, size_2=None):
    # returns the size_1-by-size_2 array with a 1 in the (pos1, pos2) position
    # if size_2 is not given, assume size_2=size_1
    if not size_2:
        size_2=size_1
    return np.array([[int(i==pos1 and j==pos2) for i in range(size_2)] for j in range(size_1)])

def full_FFN_W_QK_matrix(hidden_dimension, W_1, n_ctx, d_model, Omega=1000):
    # creates the W_QK matrix used for implementing the full output of one hidden dimension using attention
    W_QK_1=2*Omega*sum([matrix_unit(d_model+n_ctx+1, n_ctx+d_model, i+d_model) for i in range(n_ctx+1)])
    W_QK_2=2*Omega*sum([matrix_unit(d_model+n_ctx+1, i+d_model, i+d_model) for i in range(n_ctx)])
    A=np.matmul(W_1, sum([matrix_unit(W_1.shape[1], d_model+i, hidden_dimension, size_2=d_model+n_ctx+1) for i in range(n_ctx)]))
    B=np.zeros((n_ctx+1, d_model+n_ctx+1))
    W_QK_3=np.block([[A], [B]])
    W_QK=W_QK_1+W_QK_2+W_QK_3
    return W_QK

def full_FFN_W_OV_matrix(W_1, W_2, hidden_dimension, n_ctx, d_model):
    # creates the W_OV matrix used for implementing the full output of one hidden dimension using attention
    relevant_part= np.matmul(np.matmul(W_1,matrix_unit(W_1.shape[1], hidden_dimension, hidden_dimension)), W_2)
    return matrix_corner_join(relevant_part, np.zeros((n_ctx+1, n_ctx+1)))

def augment_classical_residual_stream(classical_residual_stream, zero_out_identity_matrix=False):
    # pads out a matrix by adding a row of 0s to the bottom, and a square identity matrix to the right
    # resulting matrix is (n_ctx+1)-by-(d_model+n_ctx+1) matrix
    # inputs:
    #   - classical_residual_stream - the n_ctx-by-d_model matrix to be augmented
    #   - zero_out_identity_matrix - boolean, defaults to False. If true, uses a 0 matrix instead of an identity matrix for the padding on the right
    n_ctx=classical_residual_stream.shape[0]
    d_model=classical_residual_stream.shape[1]
    zero_padding= np.zeros((1, d_model))
    identity_padding=np.identity(n_ctx+1)
    if zero_out_identity_matrix:
        identity_padding=0*identity_padding
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

def FFN_on_one_hidden_dimension(classical_residual_stream, W_1, W_2, hidden_dimension, Omega=1000):
    # creates and applies an attention head which will apply SiLU to one dimension of the residual stream
    n_ctx=classical_residual_stream.shape[0]
    d_model=classical_residual_stream.shape[1]
    augmented_residual_stream=augment_classical_residual_stream(classical_residual_stream)
    W_QK=full_FFN_W_QK_matrix(hidden_dimension, W_1, n_ctx, d_model, Omega=Omega)
    W_OV=full_FFN_W_OV_matrix(W_1, W_2, hidden_dimension, n_ctx, d_model)
    head_output=attention_head(augmented_residual_stream, W_QK, W_OV)
    return head_output

def FFN_on_all_hidden_dimensions(classical_residual_stream, W_1, W_2, Omega=1000):
    # creates and applies attention heads which apply an FFN parameterized by W_1 and W_2
    # returns the total head output (to be added to the residual stream)
    total_head_output=sum([FFN_on_one_hidden_dimension(classical_residual_stream, W_1, W_2, hidden_dimension, Omega=Omega) for hidden_dimension in range(W_1.shape[1])])
    return total_head_output

def direct_SiLU_entrywise(arr):
    # the SiLU function, vectorized to apply to each entry independently
    return np.vectorize(direct_SiLU)(arr)

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
    return head_output

def direct_FFN(residual_stream, W_1, W_2):
    # applies an FFN in the classical way
    # returns the total output of the FFN, ready to be added to the residual stream
    hidden_layers=np.matmul(residual_stream, W_1)
    hidden_layers=np.vectorize(direct_SiLU)(hidden_layers)
    return np.matmul(hidden_layers, W_2)

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

def test_FFN(test_residual_stream, W_1, W_2, Omega=1000, error_tolerance=10**-10):
    # compares result of applying an FFN:
    #   - directly
    #   - with attention heads
    # parameters:
    #   - test_residual_stream - an n_ctx-by-d_model array
    #   - Omega - the large number that overwhelms softmaxes
    #   - error_tolerance - the largest acceptable error in a single entry of the computed matrix
    attention_based_result=FFN_on_all_hidden_dimensions(test_residual_stream, W_1, W_2, Omega=Omega)
    direct_result = direct_FFN(test_residual_stream, W_1, W_2)
    augmented_direct_result=augment_classical_residual_stream(direct_result, zero_out_identity_matrix=True)
    largest_difference=np.amax(abs(attention_based_result-augmented_direct_result))
    if largest_difference<error_tolerance:
        print(f"The FFN methods had nearly identical outputs! (Max error: {largest_difference:.3e})")
    else:
        print(f"BAD! VERY LARGE DIFFERENCE IN FFN OUTPUTS! (Max error: {largest_difference:.3e})")

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
    direct_result = attention_head(test_residual_stream, test_W_QK, test_W_OV)
    augmented_direct_result=augment_classical_residual_stream(direct_result, zero_out_identity_matrix=True)
    largest_difference=np.amax(abs(attention_based_result-augmented_direct_result))
    if largest_difference<error_tolerance:
        print(f"The attention methods had nearly identical outputs! (Max error: {largest_difference:.3e})")
    else:
        print(f"BAD! VERY LARGE DIFFERENCE IN ATTENTION OUTPUTS! (Max error: {largest_difference:.3e})")

if __name__=="__main__":
    n_ctx=20
    d_model=30
    Omega=1000
    error_tolerance=10**-10
    for _ in range(100):
        test_residual_stream=np.random.normal(size=(n_ctx,d_model))
        test_W_QK=np.random.normal(size=(d_model,d_model))
        test_W_OV=np.random.normal(size=(d_model,d_model))
        test_augmented_attention(test_residual_stream, test_W_QK, test_W_OV, Omega=Omega, error_tolerance=error_tolerance)
    for _ in range(100):
        test_residual_stream=np.random.normal(size=(n_ctx,d_model))
        test_W_1=np.random.normal(size=(d_model,4*d_model))
        test_W_2=np.random.normal(size=(4*d_model,d_model))
        test_FFN(test_residual_stream, test_W_1, test_W_2, Omega=Omega, error_tolerance=error_tolerance)
