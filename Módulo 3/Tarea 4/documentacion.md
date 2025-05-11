# Sistema Experto de Diagnóstico Respiratorio
**El programa que hay es solo una demo, no la versión final**
## 1. Objetivo
Diagnosticar de manera rápida la causa **más probable** entre ocho cuadros respiratorios agudos en adultos ambulatorios, a partir de una serie de síntomas y signos seleccionados. El sistema sugiere la **acción inicial** (autocuidado, visita médica o urgencias) y explica la razón de la decisión.

### Diagnósticos cubiertos
1. Resfriado común  
2. Influenza  
3. Faringitis bacteriana  
4. Sinusitis aguda  
5. Bronquitis aguda  
6. Exacerbación asmática  
7. Exacerbación de EPOC  
8. Neumonía adquirida en la comunidad

**Fuera de alcance:** pediatría, inmunosuprimidos, COVID‑19 diferenciado, patologías crónicas raras.

---

## 2. Fuentes de conocimiento
| # | Fuente académica | Uso en el sistema |
|---|------------------|-------------------|
| 1 | Heckerling et al. *Clinical prediction rule for pneumonia* (Am Fam Physician) | Criterio de fiebre alta + tos + disnea para neumonía |
| 2 | CDC Influenza Clinical Description | Síntomas cardinales de influenza (fiebre alta, mialgias, tos seca) |
| 3 | IDSA Guideline on Acute Bacterial Rhinosinusitis | Duración >10 días + dolor facial en sinusitis bacteriana |
| 4 | Centor/McIsaac criteria (*Ann Intern Med*) | Exclusión de tos y congestión en faringitis estreptocócica |
| 5 | GOLD 2024 Report | Criterios de exacerbación de EPOC (disnea, volumen y purulencia de esputo) |
| 6 | GINA 2024 Update | Signos de exacerbación asmática (sibilancias, disnea, antecedente de asma) |
| 7 | Review: Acute Bronchitis in Adults (*BMJ Best Practice*) | Tos >5 días sin fiebre alta para bronquitis viral |
| 8 | NICE Common cold guideline | Características de resfriado común y exclusión de fiebre alta |

---

## 3. Base de hechos
| Campo | Tipo | Opciones |
|-------|------|----------|
| `fiebre_alta` | signo | si / no |
| `tos` | síntoma | seca / productiva / no |
| `tos_>5d` | duración | si / no |
| `dolor_garganta` | síntoma | si / no |
| `congestion` | síntoma | si / no |
| `dolor_facial` | síntoma | si / no |
| `dolor_muscular` | síntoma | si / no |
| `disnea` | síntoma | si / no |
| `sibilancias` | signo | si / no |
| `esputo_color` | signo | purulento / claro / no |
| `fc_alta` | signo | si / no |
| `asma_prev` | antecedente | si / no |
| `epoc_prev` | antecedente | si / no |
| `sintomas_>10d` | duración | si / no |

---

## 4. Reglas de diagnóstico (resumen)
| # | Diagnóstico | Condiciones principales | Acción |
|---|-------------|-------------------------|--------|
| R1 | Neumonía | fiebre_alta = si **y** tos (seca/productiva) **y** (disnea = si **y** fc_alta = si) | Urgencias |
| R2 | Influenza | fiebre_alta = si **y** dolor_muscular = si **y** tos = seca | Médico <24 h |
| R3 | Exacerbación asmática | asma_prev = si **y** sibilancias = si **y** disnea = si | Urgencias |
| R4 | Exacerbación EPOC | epoc_prev = si **y** disnea = si **y** esputo_color = purulento | Médico |
| R5 | Bronquitis aguda | tos = productiva **y** tos_>5d = si **y** fiebre_alta = no **y** disnea = no | Autocuidado |
| R6 | Sinusitis aguda | dolor_facial = si **y** congestion = si **y** sintomas_>10d = si | Médico |
| R7 | Faringitis bacteriana | fiebre_alta = si **y** dolor_garganta = si **y** tos = no **y** congestion = no | Médico |
| R8 | Resfriado común | congestion = si **y** dolor_garganta = si **y** fiebre_alta = no **y** disnea = no | Autocuidado |

### Formulación lógica de las reglas
A continuación se expresa cada regla diagnóstica en lógica proposicional (forma condicional).  
Las letras mayúsculas representan hechos **verdaderos**; el diagnóstico se infiere si la conjunción del antecedente es cierta.  

| Símbolo | Hecho clínico                 | Símbolo | Hecho clínico                    |
|---------|------------------------------|---------|----------------------------------|
| **F**   | fiebre alta = sí             | **Tₛ**  | tos = seca                       |
| **Tₚ**  | tos = productiva             | **D**   | disnea = sí                      |
| **FC**  | frecuencia cardíaca alta = sí| **M**   | dolor muscular = sí              |
| **Si**  | sibilancias = sí             | **A**   | antecedente de asma = sí         |
| **E**   | antecedente de EPOC = sí     | **P**   | esputo purulento = sí            |
| **TF**  | tos > 5 días = sí            | **L**   | dolor facial = sí                |
| **C**   | congestión nasal = sí        | **G**   | dolor de garganta = sí           |
| **S10** | síntomas > 10 días = sí      |


| # | Diagnóstico | Fórmula lógica |
|---|-------------|----------------|
| 1 | **Neumonía** | F ∧ (Tₛ ∨ Tₚ) ∧ (D ∨ FC) → Neumonía |
| 2 | **Influenza** | F ∧ M ∧ Tₛ → Influenza |
| 3 | **Exacerbación asmática** | A ∧ Si ∧ D → Exacerbación asmática |
| 4 | **Exacerbación de EPOC** | E ∧ D ∧ P → Exacerbación EPOC |
| 5 | **Bronquitis aguda** | Tₚ ∧ TF ∧ ¬F ∧ ¬D → Bronquitis |
| 6 | **Sinusitis aguda** | L ∧ C ∧ S10 → Sinusitis |
| 7 | **Faringitis bacteriana** | F ∧ G ∧ ¬Tₛ ∧ ¬Tₚ ∧ ¬C → Faringitis |
| 8 | **Resfriado común** | C ∧ G ∧ ¬F ∧ ¬D → Resfriado |


---

## 5. Arquitectura del sistema
```
Usuario → Interfaz Tkinter → Motor de Inferencia → Diagnóstico + Explicación
                                      ↑
                               Base de Reglas (reglas[])
                                      ↑
                               Base de Hechos (diccionario)
```
- **Motor:** algoritmo de encadenamiento hacia atrás (backward). Pregunta solo los hechos necesarios para evaluar cada regla, en el orden en que aparecen.
- **Coincidencia mínima**: una regla se activa si **todas** sus condiciones se cumplen y ninguna contradicción explícita existe.
- **Explicación:** se genera a partir de la cadena de texto asociada a la regla disparada.

---

## 6. Casos de prueba
| Caso | Entradas clave | Diagnóstico esperado |
|------|----------------|----------------------|
| A | fiebre_alta = si, tos = seca, dolor_muscular = si | Influenza |
| B | fiebre_alta = no, congestion = si, dolor_garganta = si | Resfriado común |
| C | fiebre_alta = si, tos = productiva, disnea = si | Neumonía |
| D | tos = productiva, tos_>5d = si, fiebre_alta = no, disnea = no | Bronquitis aguda |

---

## Autores 
- **Alexis Guillén Ruiz**
- **Juan Antonio Velázquez Alárcon** 
