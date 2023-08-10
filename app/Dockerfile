# syntax=docker/dockerfile:1
   
FROM python:3.8-slim
WORKDIR /app
COPY . .
RUN pip install --trusted-host pypi.python.org -r requirements.txt


# Run app.py when the container launches
ENTRYPOINT [ "python" ]
CMD ["app.py"]
EXPOSE 4000