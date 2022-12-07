<template>
  <div class="home">
    <el-row class="outer_container">
      <el-col :span="22">
        <el-row>
          <el-col :span="13">
            <Painter/>
          </el-col>
          <el-col :span="3">
            <el-button type="primary" style="top: 40vh; position: relative">Generate!</el-button>
            <el-select v-model="generatorModel" placeholder="Select" size="large" style="margin-top: 10px; top: 40vh; position: relativ">
              <el-option
                v-for="item in options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </el-col>
          <el-col :span="8" :offset="2">

          </el-col>
        </el-row>
      </el-col>
    </el-row>
    <!-- <Painter msg="Welcome to Your Vue.js App"/> -->
  </div>
</template>

<script>
// @ is an alias to /src
import Painter from '@/components/Painter.vue'

export default {
  name: 'Main',
  components: {
    Painter
  },
  data() {
    return {
      generatorModel: null,
      options: [{
        value: 0,
        label: "SPADE",
      }, {
        value: 1,
        label: "PsP encoder + StyleGANv2",
      }]
    }
  },
  computed() {

  }
}






</script>


<style scoped>
/* .outer_container {
  height: 80vh;
  align-content: center;
} */
</style>