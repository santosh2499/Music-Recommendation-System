# Stage 1: Runtime
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Entrypoint script to run both services
RUN echo '#!/bin/bash\nuvicorn main:app --host 0.0.0.0 --port 8000 & \nstreamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0' > entrypoint.sh
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
