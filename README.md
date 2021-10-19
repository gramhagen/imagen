# Imagen
Streamlit app for AI based image generation

# Deployment

## Option 1: Deploy to Azure
[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fgramhagen%2Fimagen%2Fmain%2Fdeployment%2Fmain.bicep)

## Option 2: Deploy from Azure CLI

### Using Bicep
See [Getting Started](https://github.com/Azure/bicep/tree/main#get-started-with-bicep) guide at Azure Bicep repo

```bash
RG=<RESOURCE GROUP NAME>
LOCATION=<RESOURCE GROUP LOCATION>
az group create -n $RG -l $LOCATION
az deployment group create -f ./main.bicep -g $RG -p username=$USER -p adminPasswordOrKey=$(cat ~/.ssh/id_rsa.pub)
```

# Run the application
## Connect to the app
```bash
IP=$(az vm show -d -g $RG -n ImagenVM --query "publicIps" -o tsv)
ssh $IP -L 8501:localhost:8501
# now open http://localhost:8501 in a browser
```
