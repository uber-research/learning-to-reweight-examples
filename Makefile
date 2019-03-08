PROTOC=protoc
all: cnnproto optimproto
cnnproto: models/cnn/configs/*.proto
	$(PROTOC) models/cnn/configs/*.proto --python_out .

optimproto: models/optim/*.proto
	$(PROTOC) models/optim/*.proto --python_out .

clean:
	rm -rf models/cnn/configs/*_pb2.py
	rm -rf models/optim/*_pb2.py
