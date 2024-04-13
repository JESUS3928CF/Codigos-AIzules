@REM Usando el FQDN de mi ACI
curl -X POST "http://text-analytics.erdxf7a2e7fsheh0.eastus.azurecontainer.io:5000/text/analytics/v3.0/languages" -H "Content-Type: application/json" --data-ascii "{'documents':[{'id':1,'text':'Hello world.'},{'id':2,'text':'IA control desarrollo.'}]}"
