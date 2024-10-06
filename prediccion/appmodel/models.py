from django.db import models

class Administracion(models.Model):
    rut = models.CharField(max_length=12)
    nombre = models.CharField(max_length=100)
    apellido = models.CharField(max_length=100)

class Paciente(models.Model):
    rut = models.CharField(max_length=12)
    nombre = models.CharField(max_length=100)
    apellido = models.CharField(max_length=100)
    edad = models.IntegerField()
    nacimiento = models.DateField()
    genero = models.IntegerField(choices=[(1, 'Hombre'), (0, 'Mujer'), (2, 'Otro')])
    bmi = models.DecimalField(max_digits=5, decimal_places=2)
    hipertension = models.BooleanField()
    enfermedad_cardiaca = models.BooleanField()
    nivel_hba1c = models.DecimalField(max_digits=4, decimal_places=2)
    nivel_glucosa = models.DecimalField(max_digits=5, decimal_places=2)
    historial_tabaquismo = models.IntegerField(choices=[
        (0, 'Nunca'), (1, 'Ex-fumador'), (2, 'Fumador ocasional'), (3, 'Fumador habitual')
    ])

    def __str__(self):
        return f'{self.nombre} {self.apellido}'

