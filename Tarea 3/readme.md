# Agentes Deliberativos

Los agentes deliberativos son sistemas de inteligencia artificial diseñados para tomar decisiones basadas en un modelo interno del mundo. A diferencia de los agentes reactivos, que responden directamente a estímulos del entorno, los agentes deliberativos evalúan diferentes cursos de acción antes de actuar. Para ello, emplean técnicas de razonamiento simbólico y planificación, lo que les permite alcanzar sus objetivos de manera más eficiente.  
Estos agentes son fundamentales en diversas áreas como la robótica, la automatización y la inteligencia artificial en general. En estos campos, la toma de decisiones informada y estratégica es crucial para el desempeño óptimo del sistema.

## Funcionamiento de los Agentes Deliberativos

El proceso de un agente deliberativo introduce una función de deliberación entre la percepción y la ejecución, lo que le permite seleccionar la acción más adecuada en cada momento. Este proceso se basa en dos etapas principales:

1. **Deliberación**: Consiste en definir cuáles son los objetivos que el agente desea alcanzar.
2. **Razonamiento basado en medios y fines**: Una vez definidos los objetivos, el agente analiza cuál es la mejor manera de lograrlos.

Este proceso se fundamenta en el razonamiento práctico, que implica decidir en cada momento qué acción tomar para acercarse a los objetivos establecidos.

## Ciclo de Ejecución de un Agente Deliberativo

Un ejemplo de ciclo de ejecución de un agente deliberativo podría representarse con el siguiente pseudocódigo:

```pseudo
EstadoMental s;
ColaEventos eq;
...
s.inicializa();
while (true) {
    opciones = generar_opciones (eq, s);
    seleccionado = delibera (opciones, s);
    s.actualiza_estado(seleccionado);
    ejecutar (s);
    eq.mira_eventos();
}
