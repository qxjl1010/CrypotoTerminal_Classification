/usr/bin/docker pull tensorflow/serving:latest-devel

/usr/bin/docker stop $(docker ps -aq)

/usr/bin/docker rm $(docker ps -aq)

/usr/bin/docker run -it -p 8500:8500 --name lstm_container tensorflow/serving:latest-devel

/usr/bin/docker cp /chris_ai/models lstm_container:/lstm_model

/usr/bin/docker exec -it <lstm_container> /usr/local/bin/tensorflow_model_serving --port=8500 --model_name=lstm --model_base_path=/lstm_model

/root/enter/envs/tensorflow_python3.6/bin/python /root/chris_ai/client_eval.py

/usr/bin/nc localhost 42526
