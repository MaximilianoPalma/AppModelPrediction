<script>
    document.getElementById('clearBtn').addEventListener('click', function() {
        // Limpiar el campo RUT
        document.getElementById('rut').value = '';
        
        // Ocultar la sección de resultados si existe
        var resultadoPaciente = document.getElementById('resultadoPaciente');
        if (resultadoPaciente) {
            resultadoPaciente.style.display = 'none';
        }
    });
</script>
