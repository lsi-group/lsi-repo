[Datasets](https://www.kaggle.com/datasets)
no son competiciones yo creo que para empezar está bien son datos limpitos y tareas muy claras también tienen datasets sin más sin tarea y sin anotación
[Ejemplo fraude](https://www.kaggle.com/dalpozz/creditcardfraud): esta es muy facilita es de clasificación y cuando hice cosas con ella estaba guay porque todo el trabajo a hacer es de refinado de modelos
[Ejemplo cancer](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data): esta otra también es clasificación tiene muy pocos datos, pero está guay ver si puedes generar nuevas features de las que ya existen o aplicando deeplearning puedes componerlas automáticamente
[Ejemplo churn prediction](https://www.kaggle.com/sanjaym0410/d/ludobenistant/hr-analytics/hr-analytics-complete-analysis-and-prediction): esta otra ya es una tarea un poco más compleja, tiene más chicha es una cosa que en telefonía se hace mucho que se llama churn prediction que es predecir cuando se va a dar de baja un cliente Yo creo que en cuanto saquemos cosas guays con esas nos podemos meter ya con competiciones de verdad

Dentro de cada colección, hay una sección que se llama Kernels ahí lo que hay son análisis que hace la gente de las colecciónes en notebooks  está bien leerlos antes de empezar a hacer cosas para ver cual es el trabajo de verdad de data science

Herramientas
[Databriks](https://community.cloud.databricks.com/): esta gente te da un cluster ya montado gratis (pequeño) con spark instalado además tiene varios interpretes para que programemos en Scala o Python
[MLib](http://spark.apache.org/mllib/): libreria de Machine Learning tiene implementados muchos algoritmos odemos resolver todas las tareas y así nos pegamos con ML en paralelom map-reduce, spark etc
[H20](https://h2o-release.s3.amazonaws.com/h2o/rel-shannon/22/docs-website/h2o-py/docs/index.html): Modelos más complejos (Deep Learning, GBM, ...) h2o es de más alto nivel que tensorflow y tiene implementados muchisimas redes neuronales y algoritmos avanzados. En python
[Flow](http://blog.h2o.ai/2014/11/introducing-flow/). Herramienta de H20. interfaz que te permite ver los modelos que se están ejecutando, los datasets que tienes en memoria, las evaluaciones que vas haciendo
además si no quieres "programar" puedes cargar datos, elegir modelo, aplicarlo y evaluarlo solo clickando cosas con el ratón

si nos vamos a h2o lo podemos hacer en local, que es lo más facil en databricks se puede también utilizar pero como h2o lo que te está haciendo es crearte un cluster encima de tu cluster/ordenador si lo hacemo en local tenemos más control sobre lo que hace si lo hacemos en databricks no podemos ver nada de lo que hace por debajo h2o

Competiciones KDnuggets
[Real-Time Crime Forecasting Challenge](https://nij.gov/funding/Pages/fy16-crime-forecasting-challenge.aspx): The Real-Time Crime Forecasting Challenge seeks to harness the advances in ​data science to address the challenges of cri​me and justice. It encourages data scientists across all scientific disciplines to foster innovation in forecasting methods. The goal is to develop algorithms that advance place-based crime forecasting through the use of data from one police jurisdiction.
[Parkinson](https://michaeljfox.org/ppmidatachallenge2016): Esta ya pasó la fecha, pero por si acaso queréis echarle un vistazo, está interesante.
