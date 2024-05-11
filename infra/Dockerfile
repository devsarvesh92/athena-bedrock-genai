# Use an existing base image
FROM ankane/pgvector

# Set environment variables
ENV PORT=5432
ENV POSTGRES_DB=vector_db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=mysecretpassword

# Copy initialization script to container
COPY init.sql /docker-entrypoint-initdb.d/
