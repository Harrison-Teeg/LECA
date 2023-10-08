:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. raw:: html

    <div style='clear:both'></div>
