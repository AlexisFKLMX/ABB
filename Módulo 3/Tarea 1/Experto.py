import tkinter as tk
from tkinter import ttk, messagebox

# Reglas: (nombre, accion, texto_explicacion, condiciones)
reglas = [
    (
        "neumonía",
        "acudir a urgencias",
        "fiebre alta, tos y disnea o frecuencia cardiaca alta",
        {
            "fiebre_alta": ["si"],
            "tos": ["seca", "productiva"],
            "disnea": ["si"],
            "fc_alta": ["si"],
        },
    ),
    (
        "influenza",
        "visitar al médico en <24 h",
        "fiebre alta, mialgias y tos seca",
        {
            "fiebre_alta": ["si"],
            "dolor_muscular": ["si"],
            "tos": ["seca"],
        },
    ),
    (
        "exacerbación asmática",
        "acudir a urgencias",
        "antecedente de asma, sibilancias y disnea",
        {
            "asma_prev": ["si"],
            "sibilancias": ["si"],
            "disnea": ["si"],
        },
    ),
    (
        "exacerbación de EPOC",
        "visitar al médico",
        "EPOC, disnea y esputo purulento",
        {
            "epoc_prev": ["si"],
            "disnea": ["si"],
            "esputo_color": ["purulento"],
        },
    ),
    (
        "bronquitis aguda",
        "autocuidado en casa",
        "tos >5 días productiva sin fiebre alta ni disnea",
        {
            "tos": ["productiva"],
            "tos_>5d": ["si"],
            "fiebre_alta": ["no"],
            "disnea": ["no"],
        },
    ),
    (
        "sinusitis aguda",
        "visitar al médico",
        "síntomas >10 días con dolor facial y congestión",
        {
            "dolor_facial": ["si"],
            "congestion": ["si"],
            "sintomas_>10d": ["si"],
        },
    ),
    (
        "faringitis bacteriana",
        "visitar al médico",
        "fiebre alta y dolor de garganta sin tos ni congestión",
        {
            "dolor_garganta": ["si"],
            "fiebre_alta": ["si"],
            "tos": ["no"],
            "congestion": ["no"],
        },
    ),
    (
        "resfriado común",
        "autocuidado en casa",
        "congestión y dolor de garganta sin fiebre alta ni disnea",
        {
            "congestion": ["si"],
            "dolor_garganta": ["si"],
            "fiebre_alta": ["no"],
            "disnea": ["no"],
        },
    ),
]

# Opciones válidas para cada campo
opciones = {
    "fiebre_alta": ["si", "no"],
    "tos": ["seca", "productiva", "no"],
    "tos_>5d": ["si", "no"],
    "dolor_garganta": ["si", "no"],
    "congestion": ["si", "no"],
    "dolor_facial": ["si", "no"],
    "dolor_muscular": ["si", "no"],
    "disnea": ["si", "no"],
    "sibilancias": ["si", "no"],
    "esputo_color": ["purulento", "claro", "no"],
    "fc_alta": ["si", "no"],
    "asma_prev": ["si", "no"],
    "epoc_prev": ["si", "no"],
    "sintomas_>10d": ["si", "no"],
}

# Motor de inferencia
def cumple(condiciones: dict, hechos: dict) -> bool:
    """Comprueba si los hechos satisfacen todas las condiciones."""
    return all(hechos.get(campo) in valores for campo, valores in condiciones.items())


class ExpertoGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema Experto Respiratorio")
        self.resizable(False, False)

        self.variables = {}
        fila = 0
        for campo, vals in opciones.items():
            tk.Label(self, text=campo).grid(row=fila, column=0, sticky="e", padx=5, pady=3)
            var = tk.StringVar(value="seleccione")
            self.variables[campo] = var
            combo = ttk.Combobox(
                self,
                textvariable=var,
                values=["seleccione"] + vals,
                state="readonly",
                width=15,
            )
            combo.grid(row=fila, column=1, padx=5, pady=3)
            fila += 1

        # Botones
        ttk.Button(self, text="Diagnosticar", command=self.diagnosticar).grid(
            row=fila, column=0, pady=10
        )
        ttk.Button(self, text="Borrar", command=self.borrar).grid(row=fila, column=1, pady=10)
        ttk.Button(self, text="Marcar 'No'", command=self.marcar_no).grid(
            row=fila, column=2, pady=10, padx=5
        )

    # Acciones de botones
    def diagnosticar(self):
        hechos = {}
        # Validar que todos los campos estén respondidos
        for campo, var in self.variables.items():
            valor = var.get()
            if valor == "seleccione":
                messagebox.showwarning(
                    "Datos incompletos", f"Por favor selecciona un valor para '{campo}'."
                )
                return
            hechos[campo] = valor

        # Evaluar reglas en orden
        for nombre, accion, texto_cond, condiciones in reglas:
            if cumple(condiciones, hechos):
                mensaje = (
                    f"Diagnóstico probable: {nombre.capitalize()}\n\n"
                    f"Recomendación inicial: {accion}.\n\n"
                    f"Explicación: Debido a que el paciente presenta {texto_cond}, "
                    f"el diagnóstico más probable es {nombre} y se recomienda {accion}."
                )
                messagebox.showinfo("Resultado", mensaje)
                return

        messagebox.showinfo(
            "Resultado",
            "No se pudo determinar un diagnóstico claro. Se recomienda visitar al médico.",
        )

    def borrar(self):
        """Restablece todos los campos a 'seleccione'."""
        for var in self.variables.values():
            var.set("seleccione")

    def marcar_no(self):
        """Asigna 'no' a todos los campos sin seleccionar que acepten esa opción."""
        cambios = 0
        for campo, var in self.variables.items():
            if var.get() == "seleccione" and "no" in opciones[campo]:
                var.set("no")
                cambios += 1
        if cambios == 0:
            messagebox.showinfo(
                "Marcar 'No'", "No hay campos pendientes que puedan marcarse como 'no'."
            )

if __name__ == "__main__":
    app = ExpertoGUI()
    app.mainloop()
