FROM python:3.9-slim
WORKDIR /app
COPY . .
EXPOSE 7860
RUN pip3 install -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
