# PyHeta
Large scale time series experimenting framework for data processing and deep learning models training.

## Architecture

The architecture for scaling machine learning experiments handles multiple different servers and leverages third-party software such as Relational DataBases or Message Brokers.

![alt text](https://github.com/avilarenan/PyHeta/blob/main/assets/architecture.png?raw=true)

## Code structure

It is important to notice the code structure. For example, if you want to change the model you should look at the train.py file, or if you want to change the data processing method, you should go for the dataprocessor.py.

![alt text](https://github.com/avilarenan/PyHeta/blob/main/assets/code_structure.png?raw=true)


## Setup

Suggested OS environment is Ubuntu 22.04, since this is validated. It is also suggested to run additional software such as DBs and Message Brokers in separate servers for hardware specs specialization and independence. Validated Python version is 3.12, and it is suggested to run a Python virtual environment.

1. Install requirements from within this repo working directory.

```bash
pip install requirements.txt
```

2. Have a RabbitMQ Message broker server (https://www.rabbitmq.com/).

3. Have a PostgreSQL DataBase server (https://www.postgresql.org/download/linux/ubuntu/).

It is suggested to install pgadmin (https://www.pgadmin.org/download/pgadmin-4-python/) as DB Web UI.

4. Create as many servers you want for running data processing and model training and inferecen experiments in parallel.
   
5. Update static configurations in **static_info_utils.py** and **config.py** files.

6. Create credentials.json at repository root level following the below format.

```json
{
    "SOLUTIONS_DATASERVICES_API_KEY" : "YOUR_API_KEY",
    "AMQP_USER": "USER",
    "AMQP_PASSWORD": "PASS",
    "POSTGRESQL_USER": "USER",
    "POSTGRESQL_PASS": "PASS"
}
```


## Running

1. Make sure RabbitMQ and PostgreSQL are working correctly.

2. Launch the **data processors** in a group of worker machines using the **worker_manager.py** file. 

```bash
nohup python3 worker_manager.py 1 data_worker.py
nohup python3 worker_manager.py 1 train_worker.py
```

Where the first parameter is an integer corresponding to the number of Python processes to launch by running the worker file, which is the second parameter. It depends on the hardware specs of your servers, it is suggested to start with 1 and double it until it reaches hardware capacity.

> :warning: you can run data and train workers in separate machines.
 
3. Launch the Streamlit UI.

```
streamlit run Experiments_Dashboard.py
```

4. Go to data processing launch page, choose your parameters and launch the data processing.

The progress can be tracked with processing steps outputs saved in the DB or by looking at the **logs** folder.
The RabbitMQ UI may also help to see how many instructions are running at the same time, as well as their time to finish.

5. Once the data is ready, you can launch the respective training by going to the training launch page.

You can track progress similar to the data processing launch process.

6. Final results can be analyzed in the results page.

## Acknowledgement

Thanks to Escola Politécnica da Universidade de São Paulo, and thanks to BTG Pactual for supporting the work.

<p float="left">
    <img src="https://github.com/avilarenan/PyHeta/blob/main/assets/Logo-Escola-Polit%C3%A9cnica-Minerva_Logo-Escola-Polit%C3%A9cnica-Minerva-01-scaled.jpeg" width="110">
    <img src="https://github.com/avilarenan/PyHeta/blob/main/assets/1200px-Btg-logo-blue.svg.png" width="230">
</p>