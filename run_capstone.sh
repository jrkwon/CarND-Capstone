docker run --runtime=nvidia -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --name jaerock_capstone --rm -it capstone_with_cuda9

