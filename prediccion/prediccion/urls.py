from django.contrib import admin
from django.urls import path
from appmodel import views
from django.contrib.auth.decorators import login_required

from appmodel.views import reporteinforme

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    
    path('inicio/', login_required(views.index), name='inicio'),
    
    path('listado/', login_required(views.listado_pacientes), name='listado_pacientes'),
    path('registro/', views.registro_paciente, name='registro_paciente'),
    path('consulta/', views.consulta_paciente, name='consulta_paciente'),
    
    path('editar_paciente/<int:paciente_id>/', views.editar_paciente, name='editar_paciente'),
    path('eliminar_paciente/<int:paciente_id>/', views.eliminar_paciente, name='eliminar_paciente'),   
    path('reporte_paciente/<int:paciente_id>/', views.reporte_paciente, name='reporte_paciente'), 
    path('editar_paciente_json/<int:paciente_id>/', views.editar_paciente_json, name='editar_paciente_json'),

    path('enviar_reporte/<int:paciente_id>/', views.enviar_reporte, name='enviar_reporte'),  

    path('aplicacion/', login_required(views.aplicacion), name='aplicacion'),

    path('soporte/', login_required(views.soporte), name='soporte'),
    path('registro_soporte/', views.registro_soporte, name='registro_soporte'),
    path('listado_soporte/', views.listado_soporte, name='listado_soporte'),
    
    path('editar_soporte/<int:soporte_id>/', views.editar_soporte, name='editar_soporte'),
    path('eliminar_soporte/<int:soporte_id>/', views.eliminar_soporte, name='eliminar_soporte'),
    
    path('evaluacion-riesgo/', login_required(views.evaluacion_riesgo), name='evaluacion_riesgo'),
    path('verificar-rut/', login_required(views.verificar_rut), name='verificar_rut'),

    path('descargar-reporte/', views.descargar_reporte, name='descargar_reporte'),

    path('graficos/', login_required(views.generar_graficos), name='graficos'),
    path('reportes/', reporteinforme, name='reporteinforme'),
    path('buscar-pacientes/', login_required(views.buscar_pacientes), name='buscar_pacientes'),
]
