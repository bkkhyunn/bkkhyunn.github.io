---
title: "Network"
layout: archive
permalink: /categories/network
author_profile: true
types: posts
---

{% assign posts = site.categories["network"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}