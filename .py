#st.markdown("** Visualisasi Vektor Eigen Pertama (1D)**")

import plotly.graph_objects as go           
fig_vec = go.Figure()
fig_vec.add_trace(go.Scatter(y=eigenfaces[:, 0],
mode='lines',
line=dict(color='orange', width=2),
name='Vektor Eigen #1',
hovertemplate='Index: %{x}<br>Nilai: %{y:.2f}'
))
fig_vec.update_layout(
title='Plot Vektor Eigenface Pertama',
xaxis_title='Index Piksel',
yaxis_title='Nilai',
template='plotly_white',
height=400,
margin=dict(l=40, r=40, t=50, b=30)
)
st.plotly_chart(fig_vec, use_container_width=True)