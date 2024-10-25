
from django.contrib import admin
from django.urls import path
from appmodel import views

from django.contrib.auth import views as auth_views
from appmodel.forms import CustomLoginForm
from django.contrib.auth.decorators import login_required

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    
    path('index/', login_required(views.index), name = 'index'),
    
    path('listado/', login_required(views.listado_pacientes), name = 'listado_pacientes'),

    path('registro/', views.registro_paciente, name='registro_paciente'),
    path('consulta/', views.consulta_paciente, name='consulta_paciente'),   

    path('aplicacion/', login_required(views.aplicacion), name = 'aplicacion'),
    path('informe/', login_required(views.informe), name = 'informe'),

    path('soporte/', login_required(views.soporte), name = 'soporte'),
    
    path('evaluacion-riesgo/', login_required(views.evaluacion_riesgo), name='evaluacion_riesgo'),

     # Nueva ruta para generar el reporte PDF
    path('descargar_reporte/', login_required(views.descargar_reporte), name='descargar_reporte'), 
]
