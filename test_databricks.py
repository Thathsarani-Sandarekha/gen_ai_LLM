from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.getOrCreate()

main_base_table_opera = spark.read.format("delta").load("dbfs:database_url")

main_base_table_opera.show()
