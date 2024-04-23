<h1 align="center">
  <img style="vertical-align:middle" height="200"
  src="https://github.com/AlexisGitHu/SmartMicrogrids/assets/56341573/f110e00e-92a7-4b09-9a05-b571c0f11afa">
</h1>
<p align="center">
  <i>Optimización inteligente de baterías fotovoltaicas</i>
</p>

<p align="center">
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/AlexisGitHu/SmartMicrogrids/">
        <img alt="Downloads" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
</p>

> 🚀 Repositorio de código para el TFG **Optimización inteligente de baterías fotovoltaicas**  de la rama de INSO de Alexis Gómez Chimeno.

Este repositorio contiene todo el código utilizado para hacer los desarrollos pertinentes descritos en la memoria del TFG - Optimización inteligente de baterías fotovoltaicas en entornos residenciales.

Asimismo, incluye los modelos entrenados de los agentes para que puedan ser abiertamente usados, los logs de Tensorboard, los resultados de las 1000 simulaciones por cada tipo de agente, las gráficos y los datos utilizados.

## 🛡️ Instalación
Se deben instalar los paquetes necesarios, incluidos en el archivo `requirements.txt`, en un entorno virtual de conda con el siguiente comando. 
```
pip install -r requirements.txt
```
Este código solo ha sido probado en un entorno `Linux`, dado los problemas de compatibilidad presentes en la librerías `stable_baselines3`. Dicho esto, solo se recomienda utilizar el repositorio en un S.O. `Linux`. 
## 🔥 Guía de uso

Actualmente, el código completa la POC propuesta como objetivo del TFG. No obstante, en caso de querer usar este código, basta con ejecutar desde el directorio raíz el siguiente comando.

```
python src\gym_env.py
```

Esto empezará el entrenamiento del tipo de agente, con todas sus características, especificado en el archivo `env_parameters.yaml` en la carpeta `/config`. Estos parámetros se deben ajustar a los definidos por la biblioteca `stable_baselines3` para un correcto uso.

En caso de querer variar al agente `RecurrentPPO`, también se deberá modificar el código, cambiando la función `intelligent_agent` por `recurrent_intelligent_agent`. Asimismo, las dos alternativas planteadas para la producción del estado del entorno y la capacidad  máxima de la batería también se especifican en este archivo.

