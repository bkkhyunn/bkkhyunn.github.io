---
title: "Troubleshooting on Linux"
layout: archive
permalink: /categories/troubleshooting-linux
author_profile: true
types: posts
---

{% assign posts = site.categories["troubleshooting-linux"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}