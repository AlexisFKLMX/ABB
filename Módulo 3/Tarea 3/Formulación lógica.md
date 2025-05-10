## Formulación lógica del sistema experto

### Símbolos para hechos clínicos

| Símbolo | Hecho clínico                         |
|---------|----------------------------------------|
| **F**   | fiebre alta = sí                       |
| **Tₛ**  | tos seca                               |
| **Tₚ**  | tos productiva                         |
| **D**   | disnea = sí                            |
| **FC**  | frecuencia cardíaca alta = sí          |
| **M**   | dolor muscular = sí                    |
| **Si**  | sibilancias = sí                       |
| **A**   | antecedente de asma = sí               |
| **E**   | antecedente de EPOC = sí               |
| **P**   | esputo purulento = sí                  |
| **TF**  | tos > 5 días = sí                      |
| **L**   | dolor facial = sí                      |
| **C**   | congestión nasal = sí                  |
| **G**   | dolor de garganta = sí                 |
| **S10** | síntomas > 10 días = sí                |

---

### Reglas del sistema

- **R1 → Neumonía**
- **R2 → Influenza**
- **R3 → Exacerbación asmática**
- **R4 → Exacerbación de EPOC**
- **R5 → Bronquitis aguda**
- **R6 → Sinusitis aguda**
- **R7 → Faringitis bacteriana**
- **R8 → Resfriado común**

---

### Reglas expresadas como fórmulas

- **Neumonía** | F ∧ (Tₛ ∨ Tₚ) ∧ (D ∨ FC) → **R1**  
  *Si el paciente tiene fiebre alta y tiene tos seca o tos productiva, y además tiene disnea o frecuencia cardíaca alta, entonces se sugiere el diagnóstico Neumonía (Regla 1).*

---

- **Influenza** | F ∧ M ∧ Tₛ → **R2**  
  *Si el paciente tiene fiebre alta, dolor muscular y tos seca, entonces se sugiere el diagnóstico Influenza (Regla 2).*

---

- **Exacerbación asmática** | A ∧ Si ∧ D → **R3**  
  *Si el paciente tiene antecedente de asma, sibilancias y disnea, entonces se sugiere el diagnóstico Exacerbación asmática (Regla 3).*

---

- **Exacerbación de EPOC** | E ∧ D ∧ P → **R4**  
  *Si el paciente tiene antecedente de EPOC, disnea y esputo purulento, entonces se sugiere el diagnóstico Exacerbación de EPOC (Regla 4).*

---

- **Bronquitis aguda** | Tₚ ∧ TF ∧ ¬F ∧ ¬D → **R5**  
  *Si el paciente tiene tos productiva, la tos ha durado más de 5 días, y no tiene fiebre alta ni disnea, entonces se sugiere el diagnóstico Bronquitis aguda (Regla 5).*

---

- **Sinusitis aguda** | L ∧ C ∧ S10 → **R6**  
  *Si el paciente tiene dolor facial, congestión nasal y los síntomas han durado más de 10 días, entonces se sugiere el diagnóstico Sinusitis aguda (Regla 6).*

---

- **Faringitis bacteriana** | F ∧ G ∧ ¬Tₛ ∧ ¬Tₚ ∧ ¬C → **R7**  
  *Si el paciente tiene fiebre alta y dolor de garganta, y no tiene tos seca ni tos productiva ni congestión nasal, entonces se sugiere el diagnóstico Faringitis bacteriana (Regla 7).*

---

- **Resfriado común** | C ∧ G ∧ ¬F ∧ ¬D → **R8**  
  *Si el paciente tiene congestión nasal y dolor de garganta, y no tiene fiebre alta ni disnea, entonces se sugiere el diagnóstico Resfriado común (Regla 8).*

## Autores 
- **Alexis Guillén Ruiz**
- **Juan Antonio Velázquez Alárcon** 
