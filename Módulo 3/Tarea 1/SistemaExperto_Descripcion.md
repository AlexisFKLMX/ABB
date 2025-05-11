# Sistema experto para el diagnóstico de enfermedades respiratorias en adultos ambulatorios

## Descripción del problema

En la atención ambulatoria de adultos, es común encontrar pacientes con síntomas respiratorios agudos, y no siempre se cuenta con el tiempo, conocimiento clínico o herramientas necesarias para distinguir entre cuadros como el resfriado común, la influenza o incluso una neumonía. Esta incertidumbre puede llevar a diagnósticos erróneos, visitas innecesarias a urgencias o, por el contrario, a la omisión de atención médica en casos graves.

Este tipo de situaciones representa un riesgo para la salud pública y la calidad del tratamiento, especialmente en contextos con recursos limitados o en los que el acceso a médicos está restringido. Por ello, se identificó la necesidad de desarrollar una herramienta automatizada, sencilla y rápida que brinde una orientación preliminar sobre cuál es el cuadro clínico **más probable** y la **acción inicial sugerida** (autocuidado, visita médica o acudir a urgencias).

El objetivo de este sistema experto es ayudar en la **toma de decisiones clínicas rápidas**, proporcionando explicaciones claras basadas en evidencia médica, sin reemplazar el juicio de un profesional de la salud, pero sí optimizando recursos y promoviendo decisiones informadas.

## Objetivo del sistema

El sistema experto tiene como finalidad apoyar a los usuarios en el diagnóstico preliminar de enfermedades respiratorias agudas comunes en adultos ambulatorios, identificando la **causa más probable** a partir de síntomas y signos clave. También sugiere una **acción inmediata recomendada** y proporciona una breve explicación lógica del porqué de la recomendación.

## Alcance

El sistema cubre los siguientes diagnósticos:

1. Resfriado común  
2. Influenza  
3. Faringitis bacteriana  
4. Sinusitis aguda  
5. Bronquitis aguda  
6. Exacerbación asmática  
7. Exacerbación de EPOC  
8. Neumonía adquirida en la comunidad

**No se incluyen**: pacientes pediátricos, inmunosuprimidos, patologías crónicas raras ni COVID‑19 de forma diferenciada.

## Fuentes de conocimiento

Este sistema se diseñó con base en fuentes médicas y guías clínicas reconocidas a nivel internacional, entre ellas:

- Heckerling et al. (criterios para neumonía)
- CDC (síntomas cardinales de influenza)
- IDSA (guía para sinusitis bacteriana)
- Centor/McIsaac (criterios de faringitis)
- GOLD 2024 (exacerbación de EPOC)
- GINA 2024 (exacerbación asmática)
- BMJ Best Practice (bronquitis aguda)
- NICE guidelines (resfriado común)

## Base de hechos

Los síntomas y signos considerados por el sistema incluyen:

- Fiebre alta
- Tipo de tos (seca o productiva)
- Duración de la tos
- Dolor de garganta
- Congestión nasal
- Dolor facial
- Dolor muscular
- Disnea (dificultad para respirar)
- Presencia de sibilancias
- Color del esputo
- Frecuencia cardíaca elevada
- Antecedente de asma
- Antecedente de EPOC
- Duración de síntomas >10 días

## Reglas diagnósticas

El sistema utiliza un conjunto de reglas lógicas para determinar el diagnóstico más probable. Cada regla se activa solo si **todas las condiciones se cumplen** y no hay contradicciones explícitas. A cada diagnóstico se asocia una acción sugerida:

| Diagnóstico               | Acción recomendada |
|---------------------------|--------------------|
| Neumonía                  | Ir a urgencias     |
| Influenza                 | Visita médica <24h |
| Exacerbación asmática     | Ir a urgencias     |
| Exacerbación de EPOC      | Visita médica      |
| Bronquitis aguda          | Autocuidado        |
| Sinusitis aguda           | Visita médica      |
| Faringitis bacteriana     | Visita médica      |
| Resfriado común           | Autocuidado        |

## Lógica del motor de inferencia

El sistema está basado en un **motor de inferencia con encadenamiento hacia atrás** (backward chaining), el cual:

- Analiza las reglas en orden.
- Solo solicita al usuario los hechos necesarios.
- Evalúa condiciones lógicas del tipo:  
  `fiebre_alta = sí ∧ tos = seca ∧ dolor_muscular = sí → Influenza`.

La explicación del diagnóstico se genera automáticamente con base en las condiciones que activaron la regla.

## Arquitectura del sistema

Usuario → Interfaz Tkinter → Motor de Inferencia → Diagnóstico + Explicación

↑

Base de Reglas

↑

Base de Hechos


## Casos de prueba (validación)

A continuación, algunos ejemplos que se usaron para validar el sistema:

| Caso | Entrada clave                                                  | Resultado esperado     |
|------|----------------------------------------------------------------|------------------------|
| A    | Fiebre alta, tos seca, dolor muscular                          | Influenza              |
| B    | Sin fiebre, congestión, dolor de garganta                      | Resfriado común        |
| C    | Fiebre alta, tos productiva, disnea                            | Neumonía               |
| D    | Tos productiva >5 días, sin fiebre ni disnea                   | Bronquitis aguda       |

## Conclusión

El sistema experto de diagnóstico respiratorio representa una herramienta accesible, rápida y útil para guiar la toma de decisiones preliminar ante enfermedades respiratorias comunes. No sustituye una evaluación médica, pero **agiliza la orientación inicial** y permite actuar de forma informada y responsable.

## Autores

- **Alexis Guillén Ruiz**  
- **Juan Antonio Velázquez Alarcón**
