values=["as","ds","sd","ad"]
for count, value in enumerate(values):
    print(count, value)
    print(values[count])
# Load entire dataset
X = [1,1,2,2,3,3,4,4,5,5,6]
y=[1,1,2,2,3,3,4,4,5,5,6]
# Train model
n_batches=2
for epoch in range(1):
    for i in range(n_batches):
        # Local batches and labels
        local_X, local_y = X[i*n_batches:(i+1)*n_batches], y[i*n_batches:(i+1)*n_batches]
        print(local_X,local_y)
