---
title: "CNN"
layout: archive
permalink: /categories/cnn
author_profile: true
types: posts
---

{% assign posts = site.categories["cnn"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}