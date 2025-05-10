# Reglas del Sistema Experto 

El presente documento describe el desarrollo de un **sistema experto para el diagnóstico de enfermedades respiratorias agudas frecuentes en adultos**. El sistema se basa en una serie de reglas obtenidas a partir de fuentes médicas confiables (exploradas en la tarea anterior) y tiene como objetivo orientar al usuario sobre un posible diagnóstico preliminar, así como sugerir una acción recomendada (autocuidado, consulta médica o atención de urgencia).

### Lista de síntomas, signos y factores

1. Tiene fiebre alta (**sí** / no)  
2. Tiene tos  
  - seca  
  - productiva  
  - no tiene tos  
3. Tiene dolor de garganta (**sí** / no)  
4. Tiene congestión nasal (**sí** / no)  
5. Tiene disnea (**sí** / no)  
6. Tiene dolor muscular (**sí** / no)  
7. Tiene dolor facial (**sí** / no)  
8. Tiene sibilancias (**sí** / no)  
9. Tiene esputo  
  - purulento  
  - claro  
  - no tiene esputo  
10. Tiene frecuencia cardíaca alta (**sí** / no)  
11. Tiene antecedente de asma (**sí** / no)  
12. Tiene antecedente de EPOC (**sí** / no)  
13. La tos lleva más de 5 días (**sí** / no)  
14. Los síntomas llevan más de 10 días (**sí** / no)  

---

### Neumonía

Si **tiene fiebre alta** (1),  
**tiene tos seca o productiva** (2),  
**tiene disnea** (5)  
y **tiene frecuencia cardíaca alta** (10),  
entonces el diagnóstico sugerido es **Neumonía**.  
**Recomendación:** Acudir a urgencias.

---

### Influenza

Si **tiene fiebre alta** (1),  
**tiene dolor muscular** (6)  
y **tiene tos seca** (2),  
entonces el diagnóstico sugerido es **Influenza**.  
**Recomendación:** Visitar al médico en menos de 24 h.

---

### Exacerbación asmática

Si **tiene antecedente de asma** (11),  
**tiene sibilancias** (8)  
y **tiene disnea** (5),  
entonces el diagnóstico sugerido es **Exacerbación asmática**.  
**Recomendación:** Acudir a urgencias.

---

### Exacerbación de EPOC

Si **tiene antecedente de EPOC** (12),  
**tiene disnea** (5)  
y **tiene esputo purulento** (9),  
entonces el diagnóstico sugerido es **Exacerbación de EPOC**.  
**Recomendación:** Visitar al médico.

---

### Bronquitis aguda

Si **tiene tos productiva** (2),  
**la tos lleva más de 5 días** (13),  
**no tiene fiebre alta** (1)  
y **no tiene disnea** (5),  
entonces el diagnóstico sugerido es **Bronquitis aguda**.  
**Recomendación:** Autocuidado en casa.

---

### Sinusitis aguda

Si **tiene dolor facial** (7),  
**tiene congestión nasal** (4)  
y **los síntomas llevan más de 10 días** (14),  
entonces el diagnóstico sugerido es **Sinusitis aguda**.  
**Recomendación:** Visitar al médico.

---

### Faringitis bacteriana

Si **tiene fiebre alta** (1),  
**tiene dolor de garganta** (3),  
**no tiene tos** (2)  
y **no tiene congestión nasal** (4),  
entonces el diagnóstico sugerido es **Faringitis bacteriana**.  
**Recomendación:** Visitar al médico.

---

### Resfriado común

Si **tiene congestión nasal** (4),  
**tiene dolor de garganta** (3),  
**no tiene fiebre alta** (1)  
y **no tiene disnea** (5),  
entonces el diagnóstico sugerido es **Resfriado común**.  
**Recomendación:** Autocuidado en casa.

## Autores 
- **Alexis Guillén Ruiz**
- **Juan Antonio Velázquez Alárcon** 
