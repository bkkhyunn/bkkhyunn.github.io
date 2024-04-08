---
title: "AI Math"
layout: archive
permalink: /categories/math
author_profile: true
types: posts
---

{% assign posts = site.categories["math"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}